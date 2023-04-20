clear; clc;

%% parameters
global cfg
cfg.Fs                  = 48000;                                                    % sampling freq
cfg.Nbits               = 32;                                                       % 32-bit precision
cfg.l                   = cfg.Fs * (100/1000);                                      % 100 ms individual tones = 4800 samples 
cfg.T                   = linspace(1/cfg.Fs, 0.1, cfg.l);                         % Time vector for individual tones
cfg.W                   = tukeywin(cfg.l, 20/100)';                                  % 10 ms on- and off-ramps 
cfg.L                   = cfg.Fs * 5;                                               % 5 s sequences
cfg.num_blocks          = 2;                                                        % # of blocks
cfg.num_trials          = 72;                                                      % # of trials / block
cfg.norm_factor         = 7;                                                        % normalization factor defined by max number of maskers on either side
cfg.targ_att            = 0;                                                       % 0 dB
cfg.F                   = [489 699 1000];

cfg.F_targ              = [];                                  
for i = 1:length(cfg.F)
	cfg.F_targ          = [cfg.F_targ ones(1, cfg.num_trials/length(cfg.F))*cfg.F(i)];
end
clear i

cfg.F_standard          = NaN(1, cfg.num_trials);
cfg.F_deviant           = NaN(1, cfg.num_trials);
cfg.deviant_window      = repmat(4:9, 1, 12);

cfg.standard            = repmat({'low', 'high'}, 1, cfg.num_trials/2);

cfg.code                = 63*ones(1, cfg.num_trials);
cfg.code(strcmp(cfg.standard, 'low') & (cfg.F_targ==489))   = 1;
cfg.code(strcmp(cfg.standard, 'low') & (cfg.F_targ==699))   = 2;
cfg.code(strcmp(cfg.standard, 'low') & (cfg.F_targ==1000))  = 4;
cfg.code(strcmp(cfg.standard, 'high') & (cfg.F_targ==489))  = 8;
cfg.code(strcmp(cfg.standard, 'high') & (cfg.F_targ==699))  = 16;
cfg.code(strcmp(cfg.standard, 'high') & (cfg.F_targ==1000)) = 32;

cfg.freqs                   = unique(cfg.F_targ);
cfg.chunk_size              = cfg.Fs*(500/1000);                                    % window size of attended side = 500ms
cfg.n_wndws                 = cfg.L/cfg.chunk_size;                                 % # of windowsdepends on window size and overall stimulus duration
cfg.onset_time              = 1/cfg.Fs + zeros(1, cfg.num_trials);

cfg.filename                = cell(1, cfg.num_trials);

for block = 1:cfg.num_blocks
    
    fid = fopen(['Oddball_Block_' num2str(block) '.txt'], 'w');
    
    % randomize block
    indices                 = randperm(cfg.num_trials);
    cfg.F_targ              = cfg.F_targ(indices);
    cfg.standard            = cfg.standard(indices);
    cfg.deviant_window      = cfg.deviant_window(indices);
    cfg.code                = cfg.code(indices);
    
    for trial = 1:cfg.num_trials
        
        strial	= zeros(1,cfg.L);
        if strcmp(cfg.standard{trial}, 'low')
            cfg.F_standard(trial) = cfg.F_targ(trial);
            cfg.F_deviant(trial) = cfg.F_targ(trial) + 0.05*cfg.F_targ(trial);
        else
            cfg.F_standard(trial) = cfg.F_targ(trial) + 0.05*cfg.F_targ(trial);
            cfg.F_deviant(trial) = cfg.F_targ(trial);
        end
        
        for window = 2:cfg.n_wndws % first 0.5 seconds should be silent
            if ismember(window, cfg.deviant_window(trial))
                if strcmp(cfg.standard{trial}, 'low')
                    F = cfg.F_targ(trial) + 0.05*cfg.F_targ(trial);
                else
                    F = cfg.F_targ(trial);
                end
                s(round(cfg.Fs*cfg.onset_time(trial)):round(cfg.Fs*cfg.onset_time(trial)+cfg.l-1), 1) = 10^(-cfg.targ_att/20).*(1/cfg.norm_factor).*cfg.W.*sin(2*pi*F*cfg.T);
            else
                if strcmp(cfg.standard{trial}, 'high')
                    F = cfg.F_targ(trial) + 0.05*cfg.F_targ(trial);
                else
                    F = cfg.F_targ(trial);
                end
                s(round(cfg.Fs*cfg.onset_time(trial)):round(cfg.Fs*cfg.onset_time(trial)+cfg.l-1), 1) = 10^(-cfg.targ_att/20).*(1/cfg.norm_factor).*cfg.W.*sin(2*pi*F*cfg.T);
            end
            strial((window-1)*cfg.chunk_size+1:(window-1)*cfg.chunk_size+max(length(s))) = strial((window-1)*cfg.chunk_size+1:(window-1)*cfg.chunk_size+max(length(s))) + s';
        end
        
        % create wav file
        cfg = write_trial(block, trial, strial, cfg, fid);   

        %clear temporary variables
        clear stim window s 
        
    end
    
    fclose(fid);
    save(['Oddball_Block_' num2str(block) '.mat'], 'cfg');
end


function cfg = write_trial(block, trial, strial, cfg, fid)
    savedir = fullfile(pwd, ['Oddball_Block_', num2str(block)]);
    if ~exist(savedir,'dir')
        mkdir(savedir)
    end    
    %write stim for current trial
    flnm = fullfile(savedir, ['trial_' sprintf('%1.2d',trial) '.wav']);    
    fprintf('%s\n', flnm)
    fprintf(fid, '%s%u\t%s\n', '-', cfg.code(trial), flnm);
    audio = [strial' strial']; % make stereo and then write
    audiowrite(flnm, audio, cfg.Fs, 'BitsPerSample', cfg.Nbits);
    cfg.filename{trial} = flnm; % update block metadata
end