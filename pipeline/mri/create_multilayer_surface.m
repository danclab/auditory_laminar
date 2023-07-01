function create_multilayer_surface(subject_dir, layers_n, output_dir, name_prefix, varargin)

addpath('/home/bonaiuto/Dropbox/Toolboxes/spm12');
addpath('/home/bonaiuto/Dropbox/Projects/software/MEGsurfer');

hemispheres={'lh','rh'};

output_name = strcat(name_prefix, '.ds.gii');
layers = linspace(0, 1, layers_n);
layers = layers(2:end-1);


parfor l=1:length(layers)
    layer=layers(l);
    for h=1:length(hemispheres)
        hemi=hemispheres{h};
        wm_file=fullfile(subject_dir, 'surf', sprintf('%s.white',hemi));
        out_file=fullfile(output_dir, sprintf('%s.%.1f',hemi,layer));
        [status, out]=system(sprintf('mris_expand -thickness %s %d %s', wm_file, layer, out_file))
    end
end


% Read RAS offset from freesurfer volume
ras_off_file =fullfile(subject_dir, 'mri', 'orig.mgz');
[status, out]=system(sprintf('mri_info --cras %s', ras_off_file));
cols=strsplit(out,' ')
ras_offset=[str2num(cols{1}) str2num(cols{2}) str2num(cols{3})];

% Convert freesurfer surface files to gifti
for l=1:length(layers)
    layer=layers(l);
    for h_idx=1:length(hemispheres)    
        hemi=hemispheres{h_idx};
        orig_name=fullfile(output_dir, sprintf('%s.%.1f', hemi, layer));
        new_name=fullfile(output_dir, sprintf('%s.%.1f.gii', hemi, layer));
        system(sprintf('mris_convert %s %s', orig_name, new_name));
   
        % Read in each hemisphere's gifti file and adjust for RAS offset
        g=gifti(new_name);
        % Set transformation matrix to identiy
        g.mat=eye(4);
        g=set_mat(g,'NIFTI_XFORM_UNKNOWN','NIFTI_XFORM_TALAIRACH');
        % Apply RAS offset
        g.vertices=g.vertices+repmat(ras_offset,size(g.vertices,1),1);
        save(g, new_name);
    end
    
    % combine hemispheres
    lh=fullfile(output_dir, sprintf('lh.%.1f.gii', layer));
    rh=fullfile(output_dir, sprintf('rh.%.1f.gii', layer));
    combined=fullfile(output_dir, sprintf('%.1f.gii', layer));
    combine_surfaces({lh, rh}, combined);
end

% downsample
in_surfs={fullfile(subject_dir, 'surf', 'white.gii')};
out_surfs={fullfile(output_dir, 'white.ds.gii')};
for l=1:length(layers)
    layer=layers(l);
    in_surfs{end+1}=fullfile(output_dir, sprintf('%.1f.gii', layer));
    out_surfs{end+1}=fullfile(output_dir, sprintf('%.1f.ds.gii', layer));
end

in_surfs{end+1}=fullfile(subject_dir, 'surf', 'pial.gii');
out_surfs{end+1}=fullfile(output_dir, 'pial.ds.gii');

decimate_multiple_surfaces(in_surfs, out_surfs, 0.1);

combined_name=fullfile(output_dir, output_name);
% reverse order so surface order matches electrode order in laminar recordings
out_surfs(end:-1:1) = out_surfs(:);
combine_surfaces(out_surfs, combined_name);

subj_surf_dir=fullfile(subject_dir,'surf');
    
% Compute link vectors and save in pial surface
ds_fname=fullfile(output_dir,sprintf('%s.ds.gii', name_prefix));
ds=gifti(ds_fname);

pial_fname=fullfile(subj_surf_dir,'pial.ds.gii');
pial_ds_fname=fullfile(subj_surf_dir,'pial.ds.gii');

norm=compute_surface_normals(pial_ds_fname, pial_fname, 'link_vector');
ds.normals=[];
for i=1:layers_n
    ds.normals=[ds.normals; norm];
end
ds_lv_fname=fullfile(output_dir, sprintf('%s.ds.link_vector.gii', name_prefix));
save(ds,ds_lv_fname);

pial_ds_nodeep_fname=fullfile(subj_surf_dir,'pial.ds.link_vector.nodeep.gii');
pial_ds_nodeep=gifti(pial_ds_nodeep_fname);
pial_ds_fname=fullfile(subj_surf_dir,'pial.ds.gii');
pial_ds=gifti(pial_ds_fname);
mapping=knnsearch(pial_ds.vertices,pial_ds_nodeep.vertices);
verts_to_rem=setdiff([1:size(pial_ds.vertices,1)],mapping);    
disp(verts_to_rem)


n_verts_per_layer=size(ds.vertices,1)/layers_n;
offset=0;
r=[];
for i=1:layers_n
    res = verts_to_rem + offset;
    r(end+1:end+length(res))=res;
    offset=offset+(n_verts_per_layer);
end
ds_final=remove_vertices(ds, r);

ds_lv_rm_fname=fullfile(output_dir,sprintf('%s.ds.link_vector.nodeep.gii',name_prefix));
save(ds_final,ds_lv_rm_fname);