import glob
import json
import os
import sys
import subprocess

import nibabel as nib
from mne.bem import make_flash_bem, make_scalp_surfaces


def run(subj_id, json_file):
    t1= ''
    t2= ''
    flash5_dir = ''
    flash30_dir = ''

    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)
    path = parameters["dataset_path"]
    spm_deriv_path = os.path.join(path, "derivatives", "spm", subj_id)
    sub_dirs = glob.glob(os.path.join(spm_deriv_path, "*"))
    for sub_dir in sub_dirs:
        if os.path.isdir(sub_dir):
            if 't1mprage' in sub_dir.lower():
                t1_files = glob.glob(os.path.join(sub_dir, "*.nii"))
                if len(t1_files) == 1:
                    t1=t1_files[0]
            elif 't2' in sub_dir.lower():
                t2_files = glob.glob(os.path.join(sub_dir, "*.nii"))
                if len(t2_files) == 1:
                    t2=t2_files[0]
            elif 'meflash5' in sub_dir.lower():
                flash5_dir = sub_dir
            elif 'meflash30' in sub_dir.lower():
                flash30_dir = sub_dir

    print('SUBJECT ID: {}'.format(subj_id))
    print('T1: {}'.format(t1))
    print('T2: {}'.format(t2))
    print('FLASH5: {}'.format(flash5_dir))
    print('FLASH30: {}'.format(flash30_dir))

    # Align T2 to T1
    t2_path,t2_fname=os.path.split(t2)
    t2_fname,t2_ext=os.path.splitext(t2_fname)
    t2_reg=os.path.join(t2_path,'{}_reg.{}'.format(t2_fname,t2_ext))
    cmd = ['fsl_rigid_register', '-r', t1, '-i', t2, '-o', t2_reg]
    print(' '.join(cmd))
    subprocess.run(cmd)

    # Run FS
    cmd=['recon-all','-subjid',subj_id,'-hires','-i',t1,'-all','-T2',t2_reg,'-T2pial','-expert','expert.opts','-parallel','-openmp','8']
    print(' '.join(cmd))
    subprocess.run(cmd)

    # Convert flash to mgz
    subjects_dir=os.getenv('SUBJECTS_DIR')
    mri_dir=os.path.join(subjects_dir,subj_id,'mri')
    flash_dir = os.path.join(mri_dir, "flash")
    if not os.path.exists(flash_dir):
        os.mkdir(flash_dir)
    pm_dir = os.path.join(flash_dir, 'parameter_maps')
    if not os.path.exists(pm_dir):
        os.mkdir(pm_dir)
    angle_5_imgs=[]
    flash_echos = sorted(glob.glob(os.path.join(flash5_dir,'*.nii')))
    for idx, flash_echo in enumerate(flash_echos, 1):
        flash_echo = nib.load(flash_echo)
        out_fname = os.path.join(mri_dir, 'flash', f'mef05_{idx:03d}.mgz')
        nib.save(flash_echo, out_fname)
        angle_5_imgs.append(out_fname)
    angle_30_imgs=[]
    flash_echos = sorted(glob.glob(os.path.join(flash30_dir,'*.nii')))
    for idx, flash_echo in enumerate(flash_echos, 1):
        flash_echo = nib.load(flash_echo)
        out_fname = os.path.join(mri_dir, 'flash', f'mef30_{idx:03d}.mgz')
        nib.save(flash_echo, out_fname)
        angle_30_imgs.append(out_fname)

    # Compute parameters
    tes=['1.87','3.67','5.49','7.31','9.13','10.95','12.77','14.59']
    cmd=['mri_ms_fitparms','-tr','20','-fa','5']
    for idx, te in enumerate(tes):
        cmd.extend(['-te',te,angle_5_imgs[idx]])
    cmd.extend(['-tr', '20', '-fa', '30'])
    for idx, te in enumerate(tes):
        cmd.extend(['-te', te, angle_30_imgs[idx]])
    cmd.append(pm_dir)
    print(' '.join(cmd))
    subprocess.run(cmd)

    # Create synthetic flash 5deg
    cmd = ['mri_synthesize', '20', '5', '5',
           os.path.join(pm_dir, 'T1.mgz'),
           os.path.join(pm_dir, 'PD.mgz'),
           os.path.join(pm_dir, 'flash5.mgz')
           ]
    print(' '.join(cmd))
    subprocess.run(cmd)

    # Make BEM surfaces
    make_flash_bem(subject=subj_id, subjects_dir=subjects_dir,
                   overwrite=True, show=True, copy=True,
                   register=True, flash5_img=os.path.join(pm_dir, 'flash5.mgz'), verbose=True)

    make_scalp_surfaces(subj_id, subjects_dir=subjects_dir, force=True)

if __name__=='__main__':
    try:
        subj_id = sys.argv[1]
    except:
        print("incorrect arguments")
        sys.exit()

    try:
        json_file = sys.argv[2]
        print("USING:", json_file)
    except:
        json_file = "../settings.json"
        print("USING:", json_file)


    run(subj_id, json_file)
