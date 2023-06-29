import os.path as op
import sys

import numpy as np
import mne
import json

from mne.transforms import apply_trans, invert_transform
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
from utilities import files
from mne.channels import read_dig_fif
from mne.coreg import Coregistration
from mne.io.meas_info import _empty_info

def run(index, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]
    raw_path = op.join(path, "raw")
    der_path = op.join(path, "derivatives")
    subjects_dir = op.join(der_path, 'fs')
    files.make_folder(der_path)
    proc_path = op.join(der_path, "processed")
    files.make_folder(proc_path)

    subjects = files.get_folders_files(proc_path)[0]
    subjects.sort()
    subject = subjects[index]
    subject_id = subject.split("/")[-1]

    print("ID:", subject_id)

    sub_path = op.join(proc_path, subject_id)
    files.make_folder(sub_path)

    sessions = files.get_folders(subject, 'ses', '')[2]
    sessions.sort()

    fiducial_data = pd.read_csv(op.join(proc_path, 'fiducials.csv'))

    for session in sessions:
        session_id = session.split("/")[-1]

        sess_path = op.join(sub_path, session_id)
        files.make_folder(sess_path)

        qc_folder = op.join(sess_path, "QC")
        files.make_folder(qc_folder)

        lpa_str=fiducial_data.loc[(fiducial_data['subject_id'] == subject_id) & (fiducial_data['session_id'] == session_id), 'LPA_head'].item()
        rpa_str = fiducial_data.loc[(fiducial_data['subject_id'] == subject_id) & (fiducial_data['session_id'] == session_id), 'RPA_head'].item()
        nas_str = fiducial_data.loc[(fiducial_data['subject_id'] == subject_id) & (fiducial_data['session_id'] == session_id), 'NAS_head'].item()

        lpa = np.array([float(x) for x in lpa_str.split(' ')])*1e-3
        rpa = np.array([float(x) for x in rpa_str.split(' ')])*1e-3
        nas = np.array([float(x) for x in nas_str.split(' ')])*1e-3

        dig_face_fname = op.join(raw_path, subject_id, session_id, 'meg', '{}_dig_face.stl'.format(subject_id))
        dig_face = mesh.Mesh.from_file(dig_face_fname)

        points = np.around(np.unique(dig_face.vectors.reshape([int(dig_face.vectors.size / 3), 3]), axis=0), 2) * 1e-3

        dig_montage_fname=op.join(sess_path, '{}-{}-dig.fif'.format(subject_id, session_id))
        dig = mne.channels.make_dig_montage(
            hsp=points,
            lpa=lpa,
            rpa=rpa,
            nasion=nas,
            coord_frame='head'
        )
        dig.save(dig_montage_fname,
                 overwrite=True)

        info = _empty_info(1)
        info["dig"] = read_dig_fif(fname=dig_montage_fname).dig
        info._unlocked = False

        plot_kwargs = dict(
            subject=subject_id,
            subjects_dir=subjects_dir,
            surfaces="head-dense",
            dig=True,
            eeg=[],
            meg=[],
            show_axes=True,
            coord_frame="mri",
        )
        view_kwargs = dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))

        coreg = Coregistration(info, subject_id, subjects_dir, fiducials='auto', on_defects="ignore")
        coreg._setup_digs()

        align_fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
        screenshot = align_fig.plotter.screenshot()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(screenshot, origin='upper')
        ax.set_axis_off()  # Disable axis labels and ticks
        fig.tight_layout()
        fig.savefig(op.join(qc_folder, '{}-{}-coreg-initial.png'.format(subject_id, session_id)))

        coreg.set_scale_mode('uniform')
        coreg.fit_fiducials(verbose=True)

        align_fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
        screenshot = align_fig.plotter.screenshot()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(screenshot, origin='upper')
        ax.set_axis_off()  # Disable axis labels and ticks
        fig.tight_layout()
        fig.savefig(op.join(qc_folder, '{}-{}-coreg-fit_fiducials.png'.format(subject_id, session_id)))

        coreg.fit_icp(n_iterations=50, nasion_weight=0.0,  lpa_weight=0.0, rpa_weight=0.0, verbose=True)
        align_fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
        screenshot = align_fig.plotter.screenshot()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(screenshot, origin='upper')
        ax.set_axis_off()  # Disable axis labels and ticks
        fig.tight_layout()
        fig.savefig(op.join(qc_folder, '{}-{}-coreg-fit_icp.png'.format(subject_id, session_id)))

        trans_fname=op.join(sess_path, '{}-{}-trans.fif'.format(subject_id, session_id))
        mne.write_trans(trans_fname, coreg.trans, overwrite=True)

        lpa_t=apply_trans(invert_transform(coreg.trans), lpa)*1e3
        rpa_t = apply_trans(invert_transform(coreg.trans), rpa) * 1e3
        nas_t = apply_trans(invert_transform(coreg.trans), nas) * 1e3
        lpa_t_str = ' '.join(['{}'.format(x) for x in lpa_t])
        rpa_t_str = ' '.join(['{}'.format(x) for x in rpa_t])
        nas_t_str = ' '.join(['{}'.format(x) for x in nas_t])
        fiducial_data.loc[(fiducial_data['subject_id'] == subject_id) & (fiducial_data['session_id'] == session_id), 'LPA_FS'] = lpa_t_str
        fiducial_data.loc[(fiducial_data['subject_id'] == subject_id) & (fiducial_data['session_id'] == session_id), 'RPA_FS'] = rpa_t_str
        fiducial_data.loc[(fiducial_data['subject_id'] == subject_id) & (fiducial_data['session_id'] == session_id), 'NAS_FS'] = nas_t_str

    fiducial_data.to_csv(op.join(proc_path, 'fiducials.csv'))

if __name__=='__main__':
    # parsing command line arguments
    try:
        index = int(sys.argv[1])
    except:
        print("incorrect arguments")
        sys.exit()

    try:
        json_file = sys.argv[2]
        print("USING:", json_file)
    except:
        json_file = "settings.json"
        print("USING:", json_file)

    run(index, json_file)


