import sys
import json
import numpy as np
import subprocess
import mne
import os.path as op
from utilities import files
from extra.tools import update_key_value, dump_the_dict
from ecgdetectors import Detectors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc

# Time lagged cross correlation
def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

def run(index, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    mne.set_log_level(verbose=None)

    path = parameters["dataset_path"]
    sfreq = parameters["downsample_dataset"]

    der_path = op.join(path, "derivatives")
    files.make_folder(der_path)
    proc_path = op.join(der_path, "processed")
    files.make_folder(proc_path)

    sub_path = op.join(path, "raw")
    subjects = files.get_folders_files(sub_path)[0]
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

        raw_paths = files.get_files(sess_path, "zapline-" + subject_id + "-" + session_id, "-raw.fif")[2]
        raw_paths.sort()

        event_paths = files.get_files(sess_path, subject_id + "-" + session_id, "-eve.fif")[2]
        event_paths.sort()

        ica_json_file = op.join(
            sess_path,
            "{}-{}-ICA_to_reject.json".format(subject_id, session_id)
        )

        with open(ica_json_file) as ica_file:
            ica_files = json.load(ica_file)

        ica_keys = list(ica_files.keys())
        ica_keys.sort()

        raw_ica = list(zip(raw_paths, event_paths, ica_keys))

        ecg_out = dict()
        eog_out = dict()

        eog_file_path = op.join(
            sess_path,
            "{}-{}-eog-stats.json".format(subject_id, session_id)
        )
        ecg_file_path = op.join(
            sess_path,
            "{}-{}-ecg-stats.json".format(subject_id, session_id)
        )

        ds = Detectors(sfreq)

        for (raw_path, event_path, ica_key) in raw_ica:
            ica_path = op.join(
                sess_path,
                ica_key
            )
            numero = str(raw_path.split("-")[-2]).zfill(3)

            print("INPUT RAW FILE:", raw_path)
            print("INPUT EVENT FILE:", event_path)
            print("INPUT ICA FILE:", ica_path)
            # print("INPUT EDF FILE:", edf_path)

            raw = mne.io.read_raw_fif(
                raw_path,
                verbose=False,
                preload=True
            )

            events = mne.read_events(event_path)

            ica = mne.preprocessing.read_ica(
                ica_path,
                verbose=False
            )
            raw.crop(
                tmin=raw.times[events[0, 0]],
                tmax=raw.times[events[-1, 0]]
            )
            raw.filter(1, 20)

            #raw.close()

            ica_com = ica.get_sources(raw)
            #raw = None
            gc.collect()
            ica_data = ica_com.get_data()
            ica_com.close()
            gc.collect()

            # https://github.com/berndporr/py-ecg-detectors
            # variance of the distance between detected R peaks
            # if the variance is not distinct enough from the 1 percentile,
            # signal has to be found manually, indicated as 666 in the first item
            # of the list.
            r_hr = [ds.hamilton_detector(ica_data[i]) for i in range(ica_data.shape[0])]
            r_hr_var = [np.var(np.diff(i)) for i in r_hr]
            ecg_out[ica_key] = r_hr_var
            r_hr_var = np.array(r_hr_var)

            if (np.percentile(r_hr_var, 1) - np.min(r_hr_var)) > 500:
                hr = list(np.where(r_hr_var < np.percentile(r_hr_var, 1))[0])

                for h in hr:
                    fig = plt.figure()
                    fig.suptitle("{}-{}-{}".format(subject_id, session_id, numero))
                    plt.plot(ica_data[h])
                    plt.plot(r_hr[h], ica_data[h][r_hr[h]], 'ro')
                    plt.title("Detected R peaks")
                    plt.savefig(
                        op.join(qc_folder, "{}-{}-{}-hr_comp_{}.png".format(subject_id, session_id, numero, h)),
                        dpi=150,
                        bbox_inches="tight"
                    )
                    plt.close("all")
            else:
                hr = [666]

            # find which ICs match the EOG pattern
            raw = mne.set_bipolar_reference(raw, ['EEG063'], ['EEG064'], ['EB'])

            # specify this as the eye channel
            raw.set_channel_types({'EB': 'eog'})

            ica_eog, eog_scores = ica.find_bads_eog(raw)

            # # all the numbers have to be integers
            ica_eog.extend(hr)
            ica_eog = [int(i) for i in ica_eog]

            # update of the key values
            update_key_value(ica_json_file, ica_key, ica_eog)

        # dump the stats in json files

        for i in (ecg_file_path, eog_file_path):
            if not op.exists(i):
                subprocess.run(["touch", i])

        dump_the_dict(ecg_file_path, ecg_out)
        dump_the_dict(eog_file_path, eog_out)


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