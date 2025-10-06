import shutil
import sys
import json
import os.path as op
from os import sep, remove

import mne
import numpy as np
import spm_standalone
from mne import read_epochs

from utilities import files
from lameg.util import ctf_fif_spm_conversion, batch, get_files


def get_res4(ds_path, sub, ses, run):
    ref_path = op.join(ds_path, "raw", sub, ses, "meg")
    blocks = files.get_folders_files(ref_path)[0]
    block = [i for i in blocks if "block-{}".format(run[1:]) in i][0]
    res4_path = files.get_files(block, "", ".res4")[2][0]
    return res4_path


def concat_files(file_paths, out_file):
    epochs_list = []

    # Load each epochs file and append to the list
    for fname in file_paths:
        epochs = mne.read_epochs(fname, preload=True)
        epochs_list.append(epochs)

    # Concatenate all epochs
    concatenated_epochs = mne.concatenate_epochs(epochs_list)
    concatenated_epochs.save(out_file, overwrite=True)


def diff_files(file_path1, file_path2, out_file):
    spm.spm_standalone(
        "eval",
        f"""
        spm('defaults', 'EEG');
        S=[];
        S.D='{file_path1}';
        S.outfile='{out_file}';
        D=spm_eeg_copy(S);
        D2=spm_eeg_load('{file_path2}');
        ch_idx=D.indchantype('MEG');
        D(ch_idx,:,:)=D(ch_idx,:,:)-D2(ch_idx,:,:);
        save(D);
        """,
        nargout=0
    )


def diff_conditions(subject_id, session_id, spm_path):
    out_fname = op.join(spm_path, f'spm_autoreject-{subject_id}-{session_id}-oddball_diff-epo-ave.mat')
    deviant_file = op.join(spm_path, f'spm_autoreject-{subject_id}-{session_id}-deviants-epo-ave.mat')
    standards_file = op.join(spm_path, f'spm_autoreject-{subject_id}-{session_id}-standards-epo-ave.mat')
    diff_files(deviant_file, standards_file, out_fname)

    out_fname = op.join(spm_path, f'spm_autoreject-{subject_id}-{session_id}-multitone_diff-epo-ave.mat')
    detected_file = op.join(spm_path, f'spm_autoreject-{subject_id}-{session_id}-detected-epo-ave.mat')
    undetected_file = op.join(spm_path, f'spm_autoreject-{subject_id}-{session_id}-undetected-epo-ave.mat')
    diff_files(detected_file, undetected_file, out_fname)


def merge_conditions(subject_id, session_id, ses_path):
    oddball_files=[
        op.join(ses_path, f'autoreject-{subject_id}-{session_id}-standards-epo.fif'),
        op.join(ses_path, f'autoreject-{subject_id}-{session_id}-deviants-epo.fif')
    ]
    out_fname=op.join(ses_path, f'autoreject-{subject_id}-{session_id}-oddball-epo.fif')
    concat_files(oddball_files, out_fname)

    multitone_files=[
        op.join(ses_path, f'autoreject-{subject_id}-{session_id}-detected-epo.fif'),
        op.join(ses_path, f'autoreject-{subject_id}-{session_id}-undetected-epo.fif')
    ]
    out_fname = op.join(ses_path, f'autoreject-{subject_id}-{session_id}-multitone-epo.fif')
    concat_files(multitone_files, out_fname)

    btn_files=[
        op.join(ses_path, f'autoreject-{subject_id}-{session_id}-btn_trial-epo.fif'),
        op.join(ses_path, f'autoreject-{subject_id}-{session_id}-btn_detect-epo.fif')
    ]
    out_fname = op.join(ses_path, f'autoreject-{subject_id}-{session_id}-btn-epo.fif')
    concat_files(btn_files, out_fname)


def run(index, json_file, spm):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]

    der_path = op.join(path, "derivatives")
    files.make_folder(der_path)
    proc_path = op.join(der_path, "processed")
    files.make_folder(proc_path)

    raw_meg_dir = op.join(path, "raw")
    subjects = files.get_folders_files(raw_meg_dir)[0]
    subjects.sort()
    subject = subjects[index]
    subject_id = subject.split("/")[-1]
    print("ID:", subject_id)
    subject = op.join(proc_path, subject_id)

    sessions = files.get_folders(subject, 'ses', '')[2]
    sessions.sort()

    for session in sessions:
        session_id = session.split("/")[-1]
        spm_path = op.join(session,'spm')
        files.make_folder(spm_path)

        raw_meg_path = op.join(raw_meg_dir, subject_id, session_id, "meg")
        ds_paths = files.get_folders_files(raw_meg_path)[0]
        ds_paths = [i for i in ds_paths if "misc" not in i]
        ds_paths.sort()
        res4_paths = [files.get_files(i, "", ".res4")[2][0] for i in ds_paths]
        res4_paths.sort()

        merge_conditions(subject_id, session_id, session)

        epo_types = []
        epo_paths = files.get_files(session, "autoreject", "-epo.fif")[2]
        for epo_path in epo_paths:
            epo_type=epo_path.split(sep)[-1].split("-")[5]
            if not epo_type in epo_types:
                epo_types.append(epo_type)

        for epo_type in epo_types:
            fif_paths = files.get_files(session, "autoreject", epo_type + "-epo.fif")[2]

            fif_paths.sort()

            for fif in fif_paths:
                path_split = fif.split(sep)
                filename_core = path_split[-1].split(".")[0]
                ctf_fif_spm_conversion(fif, res4_paths[0], spm_path, True, spm_instance=spm)

                average_file = op.join(session, filename_core + "-ave.fif")
                epochs = read_epochs(fif, verbose=False)
                epochs = epochs.average()
                epochs.save(average_file, overwrite=True)
                ctf_fif_spm_conversion(average_file, res4_paths[0], spm_path, False, spm_instance=spm)

        diff_conditions(subject_id, session_id, spm_path)


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

    spm = spm_standalone.initialize()

    run(index, json_file, spm)