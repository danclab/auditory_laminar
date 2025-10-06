import sys
import json
from utilities import files
import numpy as np
import os.path as op
import pandas as pd
import mne
from mne import pick_channels_regexp
from mne.io.ctf.trans import _make_ctf_coord_trans_set
from mne.transforms import apply_trans, Transform
import matplotlib.pyplot as plt

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

    qc_folder = op.join(sess_path, "QC")
    files.make_folder(qc_folder)

    dss = files.get_folders_files(meg_path)[0]
    dss = [i for i in dss if "ds" in i]
    dss.sort()

    ref_dev_head_t = None

    for ds_idx,ds in enumerate(dss):
        print("INPUT RAW FILE:", ds)
        numero = int(ds.split(".")[0][-2:])

        raw = mne.io.read_raw_ctf(
            ds,
            clean_names=True,
            verbose=False
        )
        # set reference destination from the first run
        if ref_dev_head_t is None:
            ref_dev_head_t = raw.info['dev_head_t']  # use first run's device?head transform

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
        # EDIT: rely on index rather than filename
        if (ds_idx == 0 or ds_idx == 1):
            relevant_triggers = [1,2,4,8,16,32]
            deviant_window_count = 0
            deviant_window = np.loadtxt('deviant_window_Block_{:03d}.txt'.format((ds_idx+1))) # deviant_window is an array of 2 col: deviant window & start trigger
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
        # EDIT: rely on index rather than filename
        if (ds_idx == 2 or ds_idx == 3):
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

        print(f'Duration before cropping: {raw.times[-1]}')
        raw = raw.crop(
            tmin=np.max([0,raw.times[raw_events[0, 0]] - 1.0]),
            tmax=np.min([raw.times[-1], raw.times[raw_events[-1, 0]] + 0.1])
        )
        print(f'Duration after cropping: {raw.times[-1]}')

        # Pick channels corresponding to the cHPI positions
        hpi_picks = pick_channels_regexp(raw.info['ch_names'], 'HLC00[123][123].*')

        # make sure we get 9 channels
        if len(hpi_picks) != 9:
            raise RuntimeError('Could not find all 9 cHPI channels')

        # get indices in alphabetical order
        sorted_picks = np.array(sorted(hpi_picks,
                                       key=lambda k: raw.info['ch_names'][k]))

        # make picks to match order of dig cardinial ident codes.
        # LPA (HPIC002[123]-*), NAS(HPIC001[123]-*), RPA(HPIC003[123]-*)
        hpi_picks = sorted_picks[[3, 4, 5, 0, 1, 2, 6, 7, 8]]
        del sorted_picks

        # process the entire run
        time_sl = slice(0, len(raw.times))
        chpi_data = raw[hpi_picks, time_sl][0]

        # transforms
        tmp_trans = _make_ctf_coord_trans_set(None, None)
        ctf_dev_dev_t = tmp_trans['t_ctf_dev_dev']
        del tmp_trans

        # find indices where chpi locations change (threshold is 0.00001)
        indices = [0]
        indices.extend(np.where(np.any(np.abs(np.diff(chpi_data, axis=1)) > 0.0001, axis=0))[0] + 1)
        # data in channels are in ctf device coordinates (cm)
        rrs = chpi_data[:, indices].T.reshape(len(indices), 3, 3)  # m
        # map to mne device coords
        rrs = apply_trans(ctf_dev_dev_t, rrs)
        gofs = np.ones(rrs.shape[:2])  # not encoded, set all good
        moments = np.zeros(rrs.shape)  # not encoded, set all zero
        times = raw.times[indices] + raw._first_time
        chpi_locs = dict(rrs=rrs, gofs=gofs, times=times, moments=moments)

        head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=False)

        used_coils = np.array([0, 1, 2])
        coil_labels = ['lpa', 'nas', 'rpa']

        plt.figure()

        plt.subplot(3, 1, 1)
        for idx, i in enumerate(used_coils):
            c = chpi_locs['rrs'][:, i, 0] - np.mean(chpi_locs['rrs'][:, i, 0])
            plt.plot(chpi_locs['times'], c * 1000, label=coil_labels[idx])
        plt.legend()
        # plt.ylim([-10,10])
        plt.xlim(chpi_locs['times'][[0, -1]])
        plt.ylabel('x (mm)')

        plt.subplot(3, 1, 2)
        for idx, i in enumerate(used_coils):
            c = chpi_locs['rrs'][:, i, 1] - np.mean(chpi_locs['rrs'][:, i, 1])
            plt.plot(chpi_locs['times'], c * 1000)
        # plt.ylim([-15,15])
        plt.xlim(chpi_locs['times'][[0, -1]])
        plt.ylabel('y (mm)')

        plt.subplot(3, 1, 3)
        for idx, i in enumerate(used_coils):
            c = chpi_locs['rrs'][:, i, 2] - np.mean(chpi_locs['rrs'][:, i, 2])
            plt.plot(chpi_locs['times'], c * 1000)
        # plt.ylim([-15,15])
        plt.xlim(chpi_locs['times'][[0, -1]])
        plt.ylabel('z (mm)')
        plt.xlabel('time (s)')

        plt.savefig(
            op.join(qc_folder, "{}-{}-{}-chpi.png".format(subject_id, session_id, numero)),
            dpi=150, bbox_inches="tight"
        )
        plt.close("all")

        for idx, i in enumerate(used_coils):
            sd = np.std(chpi_locs['rrs'][:, i, 0]) * 1000
            print(f'{coil_labels[idx]}, x SD={sd:.2f} mm')
        for idx, i in enumerate(used_coils):
            sd = np.std(chpi_locs['rrs'][:, i, 1]) * 1000
            print(f'{coil_labels[idx]}, y SD={sd:.2f} mm')
        for idx, i in enumerate(used_coils):
            sd = np.std(chpi_locs['rrs'][:, i, 2]) * 1000
            print(f'{coil_labels[idx]}, z SD={sd:.2f} mm')

        lpa_pos = chpi_locs['rrs'][:, used_coils[0], :]
        nas_pos = chpi_locs['rrs'][:, used_coils[1], :]
        rpa_pos = chpi_locs['rrs'][:, used_coils[2], :]

        lpa_rpa_dist = np.sqrt(np.sum((lpa_pos - rpa_pos) ** 2, axis=-1))
        lpa_nas_dist = np.sqrt(np.sum((lpa_pos - nas_pos) ** 2, axis=-1))
        rpa_nas_dist = np.sqrt(np.sum((rpa_pos - nas_pos) ** 2, axis=-1))

        plt.figure()
        plt.plot(lpa_rpa_dist, label='lpa-rpa')
        plt.plot(lpa_nas_dist, label='lpa-nas')
        plt.plot(rpa_nas_dist, label='rpa-nas')
        plt.legend()
        plt.savefig(
            op.join(qc_folder, "{}-{}-{}-chpi_dists.png".format(subject_id, session_id, numero)),
            dpi=150, bbox_inches="tight"
        )
        plt.close("all")

        print(f'LPA-RPA = {np.mean(lpa_rpa_dist) * 1000} mm')
        print(f'LPA-NAS = {np.mean(lpa_nas_dist) * 1000} mm')
        print(f'RPA-NAS = {np.mean(rpa_nas_dist) * 1000} mm')

        fig = mne.viz.plot_head_positions(head_pos, mode="traces", show=False)
        fig.savefig(
            op.join(qc_folder, "{}-{}-{}-head_pos.png".format(subject_id, session_id, numero)),
            dpi=150, bbox_inches="tight"
        )

        fig = mne.viz.plot_head_positions(
            head_pos, mode="field", destination=raw.info["dev_head_t"], info=raw.info,
            show=False
        )  # visualization 3D
        fig.savefig(
            op.join(qc_folder, "{}-{}-{}-head_pos_3d.png".format(subject_id, session_id, numero)),
            dpi=150, bbox_inches="tight"
        )

        # # Compute MRI?head transform from fiducials
        # trans = compute_trans_from_fiducials(raw, fs_ras_fids_m)
        # mne.write_trans(op.join(subjects_dir, subject_id, 'bem', f'{subject_id}-{session_id}-{numero}-trans.fif'), trans, overwrite=True)
        #
        # mne.viz.plot_alignment(raw.info, trans=trans, subject=subject_id,
        #                        subjects_dir=subjects_dir)#, surfaces=['head-dense'])
        #
        # # Estimate individualized origin
        # origin = get_individual_sphere_origin(subject_id, subjects_dir, trans)
        # origin = compute_origin_from_meg_fids(raw)
        # print(f"Using individualized SSS origin (head coords): {origin}")

        raw_sss = mne.preprocessing.maxwell_filter(
            raw, head_pos=head_pos,
            st_duration=10,
            origin=[0., 0., 0.04],
            coord_frame='head',
            destination=ref_dev_head_t,
            verbose=True
        )

        f_n = str(ds_idx+1).zfill(3)  # file number

        raw_path = op.join(
            sess_path,
            "{}-{}-{}-raw.fif".format(subject_id, session_id, f_n)
        )
        eve_path = op.join(
            sess_path,
            "{}-{}-{}-eve.fif".format(subject_id, session_id, f_n)
        )

        raw_sss, events = raw_sss.copy().resample(
            sfreq,
            npad="auto",
            events=raw_events,
            n_jobs=-1,
        )
        print(f'Duration after downsampling: {raw_sss.times[-1]}')

        raw_sss.save(
            raw_path,
            fmt="single",
            overwrite=True
        )

        print("RAW SAVED:", raw_path)

        raw_sss.close()

        mne.write_events(
            eve_path,
            events,
            overwrite=True
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