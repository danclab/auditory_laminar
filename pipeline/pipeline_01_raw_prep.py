import sys
import json
from utilities import files
import numpy as np
import os.path as op
import pandas as pd
import mne

def run(subject_idx, session_idx, json_file):
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
    subject = subjects[subject_idx]
    subject_id = subject.split("/")[-1]
    print("ID:", subject_id)
    sub_path = op.join(proc_path, subject_id)
    files.make_folder(sub_path)
    sessions = files.get_folders(subject, 'ses', '')[2]
    sessions.sort()
    session=sessions[session_idx]

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

        # -------------------------------------------------------------
        # Omit "stop MEG recording" trigger
        if raw_events[-1,2]==250:
            raw_events=raw_events[:-1,:]

        start_triggers = [1,2,4,7,8,16,32]

        # Adjust event timing based on measured delay (13 ms)
        evts_to_adjust=(raw_events[:,2] != 20) & (raw_events[:,2] != 70) & (raw_events[:,2] != 100) & (raw_events[:,2] != 252)
        raw_events[evts_to_adjust,0]=raw_events[evts_to_adjust,0]+0.013 * raw.info['sfreq']

        # Correct signal value of preceeding sample
        # I don't understand what this does
        sample_to_adjust=(raw_events[:, 1] != 0)
        raw_events[sample_to_adjust, 1] = 0

        # Add triggers (444 standards, 222 deviants, 333 undetected, 555 detected)
        tone_start_times = np.arange(0.5, 5, 0.5)
        tone_events_all = []
        for i in range(len(raw_events)):
            if raw_events[i][2] in start_triggers:
                trial_start_smpl = raw_events[i][0]
                tone_time = np.int_(np.ceil((tone_start_times * raw.info['sfreq']) + trial_start_smpl)) # standard start times in smpls
                for t in range(len(tone_time)):
                    # We had a problem with one recording where the very end was cut off, so check if event time doesnt
                    # past the recording time
                    if tone_time[t]<len(raw.times):
                        tone_events_all.append([tone_time[t],0,444])
        tone_events_all = np.array(tone_events_all)
        raw_events = np.concatenate((raw_events, tone_events_all))
        raw_events.view('i8,i8,i8').sort(order=['f1'], axis=0)

        # if oddball trial
        if (f_n == '001' or f_n == '002'):
            relevant_triggers = [1,2,4,8,16,32]
            deviant_window_count = 0
            deviant_window = np.loadtxt('deviant_window_Block_' + f_n + '.txt') # deviant_window is an array of 2 col: deviant window & start trigger
            # check if start triggers match
            for i in range(len(raw_events)):
                # is trial start
                if (raw_events[i][2] in relevant_triggers):
                    for j in range(deviant_window_count, len(deviant_window)):
                        # trial codes match
                        if (raw_events[i][2] == deviant_window[j][1]):
                            raw_events[np.int_(i + deviant_window[deviant_window_count][0] + 1)][2] = 222
                            deviant_window_count += 1
                            break
                        # trial codes don't match, participant skipped this trial, check until we get a match...
                        else:
                            deviant_window_count += 1

        # if masker & target-present trial
        if (f_n == '003' or f_n == '004'):
            relevant_triggers = [1,2,4]
            previous_relevant_trig = 0
            is_first_detected = False
            for i in range(len(raw_events)):
                current_trig = raw_events[i][2]
                if previous_relevant_trig != 0:
                    # is standard tone
                    if (previous_relevant_trig in relevant_triggers) and (current_trig == 444):
                        raw_events[i][2] = 333 # undetected
                    # is standard tone after btn press
                    elif (previous_relevant_trig == 70) and (current_trig == 444):
                        raw_events[i][2] = 555 # detected
                        if is_first_detected and (raw_events[i-2][2] == 333): # correct 2 previous samples
                            raw_events[i-2][2] = 555
                            if (raw_events[i-3][2] == 333):
                                raw_events[i-3][2] = 555
                            is_first_detected = False

                # set previous relevant trigger
                if (current_trig in relevant_triggers): # start trial
                    previous_relevant_trig = current_trig
                elif (current_trig == 70) and (previous_relevant_trig != 0): # relevant btn press
                    previous_relevant_trig = 70
                    is_first_detected = True
                elif (current_trig == 7): # target absent trial
                    previous_relevant_trig = 0

        raw_events = np.int_(raw_events)

        # Check for duplicates
        print("Checking for duplicates...")
        series = pd.Series(raw_events[:,0])
        duplicates = series.duplicated()
        for i in range(len(duplicates)):
            if (duplicates[i]):
                print("Duplicates found!")
        # -------------------------------------------------------------

        raw = raw.crop(
            tmin=np.max([0,raw.times[raw_events[0, 0]] - 1.0]),
            tmax=np.min([raw.times[-1], raw.times[raw_events[-1, 0]] + 0.1])
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