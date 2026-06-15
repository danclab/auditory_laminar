import os
import shutil
import sys
import json
import os.path as op
from os import sep
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import spm_standalone
from lameg.invert import coregister, invert_ebb, load_source_time_series
from lameg.laminar import sliding_window_model_comparison
from lameg.util import spm_context, get_fiducial_coords, load_meg_sensor_data, get_surface_names, get_bigbrain_layer_boundaries
from lameg.surf import interpolate_data
from lameg.viz import show_surface, color_map
from utilities import files
import h5py
from scipy.spatial import cKDTree, KDTree
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import mne
from mne import read_epochs


def get_roi_idx(subj_id, surf_dir, hemi, regions, surf):
    """
    Returns the vertex indices from a downsampled surface mesh that correspond
    to specified anatomical regions (from FreeSurfer annotations).

    Parameters:
    - subj_id (str): FreeSurfer subject ID.
    - surf_dir (str): Path to directory containing the postprocessed surfaces (e.g., pial.gii).
    - hemi (str or None): Hemisphere ('lh', 'rh', or None for both).
    - regions (list of str): List of region names to extract from the annotation.
    - surf (nib.GiftiImage): Downsampled surface mesh used for laminar inference.

    Returns:
    - roi_idx (np.ndarray): Array of indices into `surf` corresponding to the selected ROI.
    """
    fs_subjects_dir = os.getenv('SUBJECTS_DIR')
    fs_subject_dir = os.path.join(fs_subjects_dir, subj_id)

    roi_idx = []
    hemis = []
    if hemi is None:
        hemis.extend(['lh', 'rh'])
    else:
        hemis.append(hemi)
    for hemi in hemis:
        pial = nib.load(os.path.join(surf_dir, f'{hemi}.pial.gii'))

        annotation = os.path.join(fs_subject_dir, 'label', f'{hemi}.aparc.annot')
        label, ctab, names = nib.freesurfer.read_annot(annotation)

        name_indices = [names.index(region.encode()) for region in regions]
        orig_vts = np.where(np.isin(label, name_indices))[0]

        # Find the original vertices closest to the downsampled vertices
        kdtree = KDTree(pial.darrays[0].data[orig_vts, :])
        # Calculate the percentage of vertices retained
        dist, vert_idx = kdtree.query(surf.darrays[0].data, k=1)
        hemi_roi_idx = np.where(dist == 0)[0]
        roi_idx = np.union1d(roi_idx, hemi_roi_idx)
    return roi_idx.astype(int)


def get_cortical_thickness(multilayer_mesh, n_layers):
    """
    Estimates cortical thickness at each vertex from a multilayer surface mesh.

    Parameters:
    - multilayer_mesh (nib.GiftiImage): Combined mesh of all depth layers (equidistant).
    - n_layers (int): Total number of cortical depth surfaces in the mesh.

    Returns:
    - thickness (np.ndarray): Vertex-wise Euclidean distance between layer 1 and layer N.
    """
    verts_per_surf = int(multilayer_mesh.darrays[0].data.shape[0] / n_layers)
    thickness = np.sqrt(np.sum((multilayer_mesh.darrays[0].data[:verts_per_surf, :] - multilayer_mesh.darrays[0].data[
                                                                                      (n_layers - 1) * verts_per_surf:,
                                                                                      :]) ** 2, axis=-1))
    return thickness


def get_lead_field_rmse(gainmat_fname, n_layers, verts_per_surf):
    """
    Computes the RMSE of lead field vectors across cortical depths
    relative to the superficial (layer 1) model, for each vertex.

    Parameters:
    - gainmat_fname (str): Path to HDF5 file containing the gain matrix ('G').
    - n_layers (int): Number of cortical depth layers.
    - verts_per_surf (int): Number of vertices per layer.

    Returns:
    - rmse (np.ndarray): RMSE between deep and superficial lead fields per vertex.
    """
    with h5py.File(gainmat_fname, 'r') as file:
        lf_mat = file['G'][()]

    layer_lf_mat = np.zeros((verts_per_surf, n_layers, lf_mat.shape[1]))
    diff_layer_lf_mat = np.zeros((verts_per_surf, n_layers))
    for i in range(n_layers):
        layer_lf_mat[:, i, :] = lf_mat[i * verts_per_surf:(i + 1) * verts_per_surf, :]
    for j in range(verts_per_surf):
        for i in range(n_layers):
            diff_layer_lf_mat[j, i] = np.sqrt(np.mean((layer_lf_mat[j, i, :] - layer_lf_mat[j, 0, :]) ** 2))
    return diff_layer_lf_mat[:, -1]


def get_orientation(scalp_mesh_fname, multilayer_mesh, pial_ds_mesh, verts_per_surf):
    """
    Computes the alignment between cortical column orientation vectors and
    local scalp surface normals for each vertex.

    Parameters:
    - scalp_mesh_fname (str): Path to scalp surface (GIFTI).
    - multilayer_mesh (nib.GiftiImage): Mesh with orientation vectors per vertex.
    - pial_ds_mesh (nib.GiftiImage): Downsampled pial mesh for mapping.
    - verts_per_surf (int): Number of vertices per cortical surface.

    Returns:
    - dot_products (np.ndarray): Cosine similarity (abs dot product) between
                                 dipole and scalp normals per vertex.

    Notes:
    - Higher values indicate more radial orientations
    """
    scalp_mesh = nib.load(scalp_mesh_fname)
    scalp_vertices = scalp_mesh.darrays[1].data  # Vertex coordinates
    scalp_faces = scalp_mesh.darrays[0].data  # Face indices

    pial_ds_vertices = pial_ds_mesh.darrays[0].data  # Vertex coordinates
    multilayer_orientations = multilayer_mesh.darrays[2].data  # Orientation vectors

    # Compute surface normals
    def compute_normals(vertices, faces):
        normals = np.zeros_like(vertices)
        for i in range(faces.shape[0]):
            v0, v1, v2 = vertices[faces[i]]
            # Edge vectors
            edge1 = v1 - v0
            edge2 = v2 - v0
            # Cross product to get normal
            normal = np.cross(edge1, edge2)
            normal /= np.linalg.norm(normal)  # Normalize the normal
            normals[faces[i]] += normal
        normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize all normals
        return normals

    normals = compute_normals(scalp_vertices, scalp_faces)

    # Build a KD-tree for the scalp vertices
    tree = cKDTree(scalp_vertices)

    # Find the nearest neighbor in the scalp mesh for each vertex in the downsampled cortical mesh
    _, indices = tree.query(pial_ds_vertices)

    # Compute the dot product between the orientation and the normal vectors
    dot_products = np.abs(np.einsum('ij,ij->i', normals[indices], multilayer_orientations[:verts_per_surf, :]))
    return dot_products


def get_dist_to_scalp(scalp_mesh_fname, pial_ds_mesh):
    """
    Computes the Euclidean distance from each vertex on the cortical surface
    to the closest point on the scalp.

    Parameters:
    - scalp_mesh_fname (str): Path to scalp surface (GIFTI).
    - pial_ds_mesh (nib.GiftiImage): Downsampled pial surface.

    Returns:
    - distances (np.ndarray): Distance (in mm) from cortex to nearest scalp point per vertex.
    """
    scalp_mesh = nib.load(scalp_mesh_fname)
    scalp_vertices = scalp_mesh.darrays[1].data  # Vertex coordinates

    pial_ds_vertices = pial_ds_mesh.darrays[0].data  # Vertex coordinates

    # Build a KD-tree for the scalp vertices
    tree = cKDTree(scalp_vertices)

    # Find the nearest neighbor in the scalp mesh for each vertex in the downsampled cortical mesh
    distances, _ = tree.query(pial_ds_vertices)

    return distances


subjects = [
    'sub-104',
    'sub-105',
    'sub-106',
    'sub-114'
]
sessions = [
    'ses-01',
    'ses-01',
    'ses-01',
    'ses-01'
]

n_layers = 11


def run(epo_type, patch_size, win_size, spm):
    base_out_dir = os.path.join(
        '/home/bonaiuto/auditory_laminar/output/',
        epo_type,
        '02_mefI',
        f'patch_size_{patch_size:.2f}mm',
        f'win_size_{win_size}ms'
    )
    if not os.path.exists(base_out_dir):
        os.mkdir(base_out_dir)

    # opening a json file
    json_file = "settings.json"
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]

    der_path = op.join(path, "derivatives")
    proc_path = op.join(der_path, "processed")

    for subject_id, session_id in zip(subjects, sessions):
        print(f"ID:{subject_id}, SESSION: {session_id}")

        sub_path = op.join(proc_path, subject_id)
        fid_name = op.join(proc_path, 'fiducials.csv')
        nas, lpa, rpa = get_fiducial_coords(subject_id, fid_name, col_delimiter=',', subject_column='subject_id',
                                            nas_column='NAS_MRI', lpa_column='LPA_MRI', rpa_column='RPA_MRI',
                                            val_delimiter=' ')

        print(f'NAS: {nas}')
        print(f'LPA: {lpa}')
        print(f'RPA: {rpa}')

        # Native space MRI to use for coregistration
        mri_path = op.join(
            der_path,
            'spm',
            subject_id
        )
        t1_path = files.get_folders(mri_path, 'SAGT1MPRAGE', '')[1][0]

        mri_fname = files.get_files(t1_path, 's', '.nii')[2][0]

        surf_path = op.join(
            sub_path,
            'surf'
        )
        pial_fname = op.join(surf_path, 'pial.ds.link_vector.fixed.gii')
        ds_pial = nib.load(pial_fname)

        multilayer_mesh_fname = os.path.join(surf_path, 'multilayer.11.ds.link_vector.fixed.gii')
        multilayer_mesh = nib.load(multilayer_mesh_fname)
        n_layers = 11
        verts_per_surf = int(multilayer_mesh.darrays[0].data.shape[0] / n_layers)

        layer_fnames = get_surface_names(
            n_layers,
            surf_path,
            'link_vector.fixed'
        )

        sess_path = op.join(sub_path, session_id)
        spm_path = op.join(sess_path, 'spm')

        out_path = op.join(sess_path, 'inverted')
        files.make_folder(out_path)

        localizer_woi = [15 + 2000, 55 + 2000]  # -250, 250 accounting for 2000ms offset

        epo = op.join(spm_path, f'spm_autoreject-{subject_id}-{session_id}-{epo_type}-epo-ave.mat')
        # Extract base name and path of data file
        data_path, data_file_name = op.split(epo)
        data_base = op.splitext(data_file_name)[0]

        shutil.copy(
            op.join(data_path, f'{data_base}.mat'),
            op.join(out_path, f'{data_base}.mat')
        )
        shutil.copy(
            op.join(data_path, f'{data_base}.dat'),
            op.join(out_path, f'{data_base}.dat')
        )

        base_fname = op.join(out_path, f'{data_base}.mat')

        epochs = read_epochs(op.join(sess_path, f'autoreject-{subject_id}-{session_id}-{epo_type}-epo.fif'),
                             verbose=False, preload=True)
        rank = mne.compute_rank(epochs, rank='info')

        # Number of temporal modes to use for EBB inversion
        n_temp_modes = 4

        # Coregister data to multilayer mesh
        coregister(
            nas,
            lpa,
            rpa,
            mri_fname,
            multilayer_mesh_fname,
            base_fname,
            spm_instance=spm,
            viz=True
        )

        [_, _, MU] = invert_ebb(
            multilayer_mesh_fname,
            base_fname,
            n_layers,
            woi=localizer_woi,
            patch_size=patch_size,
            n_temp_modes=n_temp_modes,
            n_spatial_modes=rank['mag'],
            return_mu_matrix=True,
            spm_instance=spm,
            viz=True
        )

        roi_idx = get_roi_idx(subject_id, surf_path, 'lh', ['postcentral'], ds_pial)

        layer_ts, ts_time, _ = load_source_time_series(
            base_fname,
            mu_matrix=MU,
            vertices=(5 - 1) * verts_per_surf + np.arange(verts_per_surf)
        )
        ts_time = ts_time - 2000
        # #t_idx = np.where((ts_time >= -500) & (ts_time <= -50))[0]
        # #t_idx = np.where((ts_time >= -65) & (ts_time <= -35))[0]
        # t_idx = np.where(ts_time <= -50)[0]
        # # signal_mag=np.max(np.abs(all_layer_ts[:,t_idx]),axis=-1)
        # #signal_mag = np.var(all_layer_ts[:, t_idx], axis=-1)
        # signal_mag = np.max(np.abs(all_layer_ts[:, t_idx]), axis=-1)

        base_t_idx = np.where((ts_time >= -2000) & (ts_time <= -1000))[0]
        m_base = np.mean(np.abs(layer_ts[:, base_t_idx]), axis=-1)
        t_idx = np.where((ts_time >= 15) & (ts_time <= 55))[0]
        signal_mag = np.mean(np.abs(layer_ts[:, t_idx]), axis=-1) - m_base

        thickness = get_cortical_thickness(multilayer_mesh, n_layers)
        gainmat_fname = os.path.join(out_path, f'SPMgainmatrix_{data_base}_1.mat')
        lf_rmse = get_lead_field_rmse(gainmat_fname, n_layers, verts_per_surf)
        scalp_mesh_fname = os.path.join(der_path, 'spm', subject_id, 'scalp_2562.surf.gii')
        orientations = get_orientation(scalp_mesh_fname, multilayer_mesh, ds_pial, verts_per_surf)
        distances = get_dist_to_scalp(scalp_mesh_fname, ds_pial)

        # Z-score each anatomical predictor (note: invert distance to scalp)
        z_thickness = zscore(thickness)
        z_lf_rmse = zscore(lf_rmse)
        z_orient = zscore(orientations)
        z_inv_dist = zscore(-distances)

        # Composite anatomical score
        anatomical_score = z_thickness + z_lf_rmse + z_orient + z_inv_dist

        # Restrict to roi_idx
        roi_anat_score = anatomical_score[roi_idx]
        roi_signal = signal_mag[roi_idx]

        # Select top 1% signal vertices within ROI
        signal_threshold = np.percentile(roi_signal, 99)
        high_signal_mask = roi_signal >= signal_threshold
        candidate_indices = roi_idx[high_signal_mask]

        # Among them, select the vertex with the best anatomical suitability
        candidate_anat_scores = anatomical_score[candidate_indices]
        best_idx_local = np.argmax(candidate_anat_scores)
        prior = candidate_indices[best_idx_local]

        prior_ts = layer_ts[prior, :]

        # Run sliding time window model comparison
        n_t = 4
        # Need to use fewer temporal modes with a smaller window size
        if win_size == 10:
            n_t = 1
        [Fs, wois] = sliding_window_model_comparison(
            prior,
            nas,
            lpa,
            rpa,
            mri_fname,
            layer_fnames,
            base_fname,
            spm_instance=spm,
            viz=False,
            invert_kwargs={
                'patch_size': patch_size,
                'n_temp_modes': n_t,
                'n_spatial_modes': rank['mag'],
                'win_size': win_size,
                'win_overlap': True
            }
        )

        # Save results
        out_fname = os.path.join(base_out_dir, f'results_{subject_id}_{session_id}.npz')
        np.savez(
            out_fname,
            subject=subject_id,
            session=session_id,
            epoch=epo_type,
            prior=prior,
            prior_ts=prior_ts,
            ts_time=ts_time,
            Fs=Fs,
            wois=wois-2000
        )

if __name__ == '__main__':
    spm = spm_standalone.initialize()

    for epo_type in ['btn_trial']:
        for patch_size in [2.5, 5.0, 7.5, 10.0]:
            for win_size in [10, 25, 50]:
                run(epo_type, patch_size, win_size, spm)

    spm.terminate()