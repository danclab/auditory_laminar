import sys
import json
import os.path as op
from os import sep
import pandas as pd
import numpy as np
from mne import read_epochs, read_events, set_log_level, pick_channels
from utilities import files
from autoreject import AutoReject
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

set_log_level(verbose=False)

def run(index, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]

    der_path = op.join(path, "derivatives")
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

    for session in sessions:
        session_id = session.split("/")[-1]

        sess_path = op.join(sub_path, session_id)
        files.make_folder(sess_path)

        qc_folder = op.join(sess_path, "QC")
        files.make_folder(qc_folder)

        epo_paths = files.get_files(sess_path, "sub", "-epo.fif")[2]

        epo_paths.sort()
        eve_paths = files.get_files(sess_path, "sub", "-eve.fif")[2]
        eve_paths.sort()

        for epo in epo_paths:
            # for epo in [epo_paths[0]]:
            numero = epo.split(sep)[-1].split("-")[4]
            epo_type = epo.split(sep)[-1].split("-")[5]

            eve_path = [i for i in eve_paths if numero + '-eve' in i][0]

            print("EVE:", eve_path.split(sep)[-1])
            print("EPO:", epo.split(sep)[-1])

            epochs = read_epochs(epo, verbose=False, preload=True)
            print("AMOUNT OF EPOCHS:", len(epochs))

            name = "{}-{}-{}-{}".format(subject_id, session_id, numero, epo_type)

            fig = epochs.average().plot(spatial_colors=True, show=False)
            plt.savefig(op.join(qc_folder, "{}-pre-autorej_erf.png".format(name)))
            plt.close("all")

            ar = AutoReject(
                consensus=np.linspace(0, 1.0, 27),
                n_interpolate=np.array([1, 4, 32]),
                thresh_method="bayesian_optimization",
                cv=10,
                n_jobs=-1,
                random_state=42,
                verbose="progressbar"
            )
            ar.fit(epochs)

            ar_fname = op.join(
                qc_folder,
                "{}-autoreject.h5".format(name)
            )
            ar.save(ar_fname, overwrite=True)
            epochs_ar, rej_log = ar.transform(epochs, return_log=True)

            rej_log.plot(show=False)
            plt.savefig(op.join(qc_folder, "{}-autoreject-log.png".format(name)))
            plt.close("all")

            epochs_ar.average().plot(spatial_colors=True, show=False)
            plt.savefig(op.join(qc_folder, "{}-post-autorej_erf.png".format(name)))
            plt.close("all")

            cleaned = op.join(sess_path, "autoreject-" + epo.split(sep)[-1])
            epochs_ar.save(
                cleaned,
                overwrite=True
            )
            print("CLEANED EPOCHS SAVED:", cleaned)


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