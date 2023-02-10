import sys
import json
from utilities import files
import numpy as np
import os.path as op
import mne

def run(index, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)
    path = parameters["dataset_path"]
    sfreq = parameters["downsample_dataset"]
    sub_path = op.join(path, "raw")
    der_path = op.join(path, "derivatives")
    files.make_folder(der_path)
    proc_path = op.join(der_path, "processed")
    files.make_folder(proc_path)
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

        meg_path = op.join(session, "meg")

        sess_path = op.join(sub_path, session_id)
        files.make_folder(sess_path)

        dss = files.get_folders_files(meg_path)[0]
        dss = [i for i in dss if "ds" in i]
        dss.sort()

        for ds in dss:
            print("INPUT RAW FILE:", ds)
            numero = int(ds.split(".")[0][-2:])
            f_n = str(numero).zfill(3)  # file number

            raw = mne.io.read_raw_ctf(
                ds,
                clean_names=True,
                verbose=False
            )
            raw_events = mne.find_events(
                raw,
                stim_channel="UPPT002",
                min_duration=0.002,
                verbose="DEBUG",
                consecutive=True
            )
            if raw_events[-1,2]==250:
                raw_events=raw_events[:-1,:]

            # Adjust event timing based on measured delay
            evts_to_adjust=(raw_events[:,2]!=70) & (raw_events[:,2]!=100)
            raw_events[evts_to_adjust,0]=raw_events[evts_to_adjust,0]+0.013 * raw.info['sfreq']

            raw = raw.crop(
                tmin=np.max([0,raw.times[raw_events[0, 0]] - 1.0]),
                tmax=np.min([raw.times[-1], raw.times[raw_events[-1, 0]] + 1.0])
            )

            raw, events = raw.copy().resample(
                sfreq,
                npad="auto",
                events=raw_events,
                n_jobs=-1,
            )

            f_n = str(numero).zfill(3)  # file number

            raw_path = op.join(
                sess_path,
                "{}-{}-{}-raw.fif".format(subject_id, session_id, f_n)
            )
            eve_path = op.join(
                sess_path,
                "{}-{}-{}-eve.fif".format(subject_id, session_id, f_n)
            )

            raw.save(
                raw_path,
                fmt="single",
                overwrite=True)

            print("RAW SAVED:", raw_path)

            raw.close()

            mne.write_events(
                eve_path,
                events
            )

            print("EVENTS SAVED:", eve_path)

if __name__=='__main__':
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