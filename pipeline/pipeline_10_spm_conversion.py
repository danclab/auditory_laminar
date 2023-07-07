import sys
import json
import os.path as op
from os import sep, remove

from mne import read_epochs

from utilities import files
import matlab.engine
from os import sep

def get_res4(ds_path, sub, ses, run):
    ref_path = op.join(ds_path, "raw", sub, ses, "meg")
    blocks = files.get_folders_files(ref_path)[0]
    block = [i for i in blocks if "block-{}".format(run[1:]) in i][0]
    res4_path = files.get_files(block, "", ".res4")[2][0]
    return res4_path

def average_filter_convert(file_path, ds_path, parasite, filt=False, l_freq=None, h_freq=None):
    path_split = file_path.split(sep)
    filename_core = path_split[-1].split(".")[0]
    sub = filename_core[11:18]
    ses = filename_core[19:25]
    run = filename_core[26:29]
    res4_path = get_res4(ds_path, sub, ses, run)
    dir_path = str(sep).join(path_split[:-1] + ["avg_spm", ""])
    files.make_folder(dir_path)

    filt_status = "_no_filter"
    if filt:
        filt_status = "_filt"

    output_file = "spm-converted{}_{}".format(filt_status, filename_core)
    output_path = op.join(dir_path, output_file)
    average_file = output_path + "-ave.fif"
    mat_output = output_path + ".mat"
    if not op.exists(mat_output):
        if not op.exists(average_file):
            epochs = read_epochs(file_path, verbose=False)
            epochs = epochs.average()
            if filt:
                epochs.filter(l_freq=l_freq, h_freq=h_freq)
            epochs.save(average_file)

        parasite.convert_mne_to_spm(res4_path, average_file, mat_output, 0, nargout=0)
        if op.isfile(average_file):
            remove(average_file)
        else:
            print(average_file, "does not exists")
        print(filename_core, "converted")

    else:
        print(mat_output, "exists")

    return mat_output


def run(index, json_file, parasite):
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

    raw_meg_dir = op.join(path, "raw")

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

        #### MODIFY THE FIF SEARCH PATHS ####

        epo_paths = files.get_files(session, subject_id + "-" + session_id + "-001", "-epo.fif")[2]
        epo_types = []
        for epo in epo_paths:
            epo_types.append(epo.split(sep)[-1].split("-")[5])

        for epo_type in epo_types:
            fif_paths = files.get_files(session, "autoreject-sub", epo_type + "-epo.fif")[2]

            fif_paths.sort()

            fif_res4_paths = list(zip(fif_paths, res4_paths))
            for fif, res4 in fif_res4_paths:
                path_split = fif.split(sep)
                filename_core = path_split[-1].split(".")[0]

                output_file = "spm-converted_{}".format(filename_core)
                output_path = op.join(spm_path, output_file)
                average_file =  op.join(session, filename_core + "-ave.fif")
                mat_output = output_path + ".mat"

                epochs = read_epochs(fif, verbose=False)
                epochs = epochs.average()
                epochs.save(average_file)

                parasite.convert_mne_to_spm(res4, average_file, mat_output, 0, nargout=0)


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

    parasite = matlab.engine.start_matlab()

    run(index, json_file, parasite)