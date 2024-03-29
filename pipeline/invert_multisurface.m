function out_file=invert_multisurface(output_dir, fiducials_csv,...
    mat_file, t1_file, surf_fname, mu_file, it_file,...
    json_out_file, layers, sub, ses, run, epo)

addpath('/home/mszul/git/DANC_spm12/spm12')
spm('defaults','eeg');
spm_jobman('initcfg');

set(0,'DefaultFigureVisible','off')

patch_size=5;
n_temp_modes=4;

subj_info=readtable(fiducials_csv,'Delimiter',',');
s_idx=find((strcmp(cellstr(subj_info.subject_id),sub)) & (strcmp(cellstr(subj_info.session_id),ses)));
nas=cellfun(@str2num,split(subj_info.NAS_MRI{s_idx},' '));
lpa=cellfun(@str2num,split(subj_info.LPA_MRI{s_idx},' '));
rpa=cellfun(@str2num,split(subj_info.RPA_MRI{s_idx},' '));

% Data file to load
data_file=mat_file
mri_fname=t1_file;
    
invert_multisurface_results=[];
invert_multisurface_results.subj_id=sub;
invert_multisurface_results.session_id=ses;
invert_multisurface_results.run_id=run;
invert_multisurface_results.epo=epo;
invert_multisurface_results.patch_size=patch_size;
invert_multisurface_results.n_temp_modes=n_temp_modes;


invert_multisurface_results.surf_fname=surf_fname;
    
% Create smoothed meshes
[smoothkern]=spm_eeg_smoothmesh_multilayer_mm(surf_fname, patch_size, layers);

% Coregistered filename
[path,base,ext]=fileparts(mat_file);
coreg_fname=fullfile(output_dir, sprintf('multilayer_%s.mat', base));
 
res_woi=[-Inf Inf];

% Run inversion
out_file=invert_ebb(data_file, coreg_fname, mri_fname, surf_fname,...
    nas, lpa, rpa, patch_size, n_temp_modes, [-Inf Inf], res_woi);
invert_multisurface_results.res_surf_fname=out_file;

% Load mesh results
D=spm_eeg_load(coreg_fname);
M=D.inv{1}.inverse.M;
U=D.inv{1}.inverse.U{1};
MU=M*U;
It=D.inv{1}.inverse.It;

dlmwrite(mu_file, MU, '\t');
invert_multisurface_results.mu_fname=mu_file;

dlmwrite(it_file, It, '\t');
invert_multisurface_results.it_fname=it_file;

fid = fopen(json_out_file,'w');
fwrite(fid, jsonencode(invert_multisurface_results)); 
fclose(fid); 
