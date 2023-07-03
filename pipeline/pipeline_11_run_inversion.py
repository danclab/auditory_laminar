import sys
import json
import os.path as op
from os import sep
import numpy as np
from os import sep, remove
from mne import read_epochs, set_log_level
from utilities import files
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pylab as plt
import matlab.engine

set_log_level(verbose=False)



def invert_multisurface(inverted_output, t1_file, mat_file, subjects_info, parasite, layers=11):
    """
    inverted_output
    """
    files.make_folder(inverted_output)
    input_path = Path(mat_file)
    bits = input_path.name.split("_")[-1].split("-")
    sub = "-".join([bits[1], bits[2]])
    epo = bits[6]
    ses = "-".join([bits[3], bits[4]])
    run = bits[5]
    link_vector = files.get_files(
        op.join(Path(inverted_output).parent),
        "multilayer", ".ds.link_vector.nodeep.gii"
    )[2][0]
    mu_file = op.join(inverted_output, "MU_" + input_path.stem + ".tsv")
    it_file = op.join(inverted_output, "IT_" + input_path.stem + ".tsv")
    res_file = op.join(inverted_output, "res_" + input_path.stem + ".tsv")
    json_out_file = op.join(inverted_output, "invert-res_" + input_path.stem + ".json")
    parasite.invert_multisurface(
        inverted_output, subjects_info, mat_file, t1_file,
        link_vector, mu_file, it_file, res_file,
        json_out_file, str(input_path.stem), float(layers), sub, ses, run, epo
    )

def run(subject_index, session_idx, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]

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
    session=sessions[session_idx]
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
