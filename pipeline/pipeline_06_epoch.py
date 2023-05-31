import sys
import json
import mne
import os.path as op
import numpy as np

from utilities import files

def run(subject_index, session_index, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]
    hi_pass = parameters["hi_pass_filter"]

    der_path = op.join(path, "derivatives")
    files.make_folder(der_path)
    proc_path = op.join(der_path, "processed")
    files.make_folder(proc_path)

    sub_path = op.join(path, "raw")
    subjects = files.get_folders_files(sub_path)[0]
    subjects.sort()
    subject = subjects[subject_index]
    subject_id = subject.split("/")[-1]

    print("ID:", subject_id)

    sub_path = op.join(proc_path, subject_id)
    files.make_folder(sub_path)

    sessions = files.get_folders(subject, 'ses', '')[2]
    sessions.sort()
    session=sessions[session_index]
    session_id = session.split("/")[-1]

    sess_path = op.join(sub_path, session_id)
    files.make_folder(sess_path)

    qc_folder = op.join(sess_path, "QC")
    files.make_folder(qc_folder)

    raw_paths = files.get_files(sess_path, "zapline-" + subject_id + "-" + session_id, "-raw.fif")[2]
    raw_paths.sort()

    ica_json_file = op.join(
        sess_path,
        "{}-{}-ICA_to_reject.json".format(subject_id, session_id)
    )

    with open(ica_json_file) as ica_file:
        ica_files = json.load(ica_file)

    ica_keys = list(ica_files.keys())
    ica_keys.sort()

    event_paths = files.get_files(sess_path, subject_id + "-" + session_id, "-eve.fif")[2]
    event_paths.sort()

    raw_ica_eve = list(zip(raw_paths, ica_keys, event_paths))

    for (raw_path, ica_key, eve_path) in raw_ica_eve:
        # for (raw_path, ica_key, eve_path) in [raw_ica_eve[3]]:
        ica_path = op.join(
            sess_path,
            ica_key
        )
        numero = str(raw_path.split("-")[-2]).zfill(3)

        print("INPUT RAW FILE:", raw_path)
        print("INPUT EVENT FILE:", eve_path)
        print("INPUT ICA FILE:", ica_path)

        ica_exc = ica_files[ica_key]

        events = mne.read_events(eve_path)

        ica = mne.preprocessing.read_ica(
            ica_path,
            verbose=False
        )

        raw = mne.io.read_raw_fif(
            raw_path,
            verbose=False,
            preload=True
        )

        raw = ica.apply(
            raw,
            exclude=ica_exc,
            verbose=False
        )
        raw = raw.pick_types(meg=True, eeg=False, ref_meg=True)

        raw.filter(
            l_freq=None,
            h_freq=hi_pass
        )

        # -------------------------------------------------------------
        # Lets keep the button presses in there so we can compare the laminar results with our other datasets
        epochs_dict = {
            "btn_trial": [[100], -.5, .5, -.5, -.3]
        }
        if len(np.where(events[:, -1] == 70)[0]):
            epochs_dict["btn_detect"] = [[70], -.5, .5, -.5, -.3]

        # masker trial
        if len(np.where(events[:,-1]==70)[0]):
            epochs_dict["undetected"]=[[333], -.05, .45, -.05, -.025] # triggers, epoch time period, basline time period
            epochs_dict["detected"]=[[555], -.05, .45, -.05, -.025]
        # oddball trial
        elif len(np.where(events[:,-1]==222)[0]):
            epochs_dict["standards"]=[[444], -.05, .45, -.05, -.025]
            epochs_dict["deviants"]=[[222], -.05, .45, -.05, -.025]
        # -------------------------------------------------------------

        for i in epochs_dict.keys():
            trig, tmin, tmax, bmin, bmax = epochs_dict[i]
            epoch = mne.Epochs(
                raw,
                mne.pick_events(events, include=trig),
                tmin=tmin,
                tmax=tmax,
                baseline=(bmin, bmax),
                verbose=True,
                detrend=1
            )

            epoch_path = op.join(
                sess_path,
                "{}-{}-{}-{}-epo.fif".format(subject_id, session_id, numero, i)
            )

            epoch.save(
                epoch_path,
                fmt="double",
                overwrite=True,
                verbose=False,
            )



if __name__=='__main__':
    # parsing command line arguments
    try:
        subj_index = int(sys.argv[1])
    except:
        print("incorrect arguments")
        sys.exit()

    try:
        sess_index = int(sys.argv[2])
    except:
        print("incorrect arguments")
        sys.exit()

    try:
        json_file = sys.argv[3]
        print("USING:", json_file)
    except:
        json_file = "settings.json"
        print("USING:", json_file)


    run(subj_index, sess_index, json_file)