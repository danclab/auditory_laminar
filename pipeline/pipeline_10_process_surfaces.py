import os.path as op
import json
import sys

from lameg.surf import postprocess_freesurfer_surfaces

from utilities import files

def run(index, json_file):
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

    postprocess_freesurfer_surfaces(
        subject_id,
        op.join(subject, 'surf'),
        'multilayer.11.ds.link_vector.fixed.gii',
        n_surfaces=11,
        ds_factor=0.1,
        orientation='link_vector',
        remove_deep=True
    )


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