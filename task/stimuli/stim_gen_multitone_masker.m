clear all; clc;

%% parameters
global cfg
cfg.Fs          = 48000;                                                        % sampling freq
cfg.Nbits       = 32;                                                           % 32-bit precision
cfg.tnln        = 100;                                                          % individual tone length, in ms
cfg.l           = floor(cfg.Fs * (cfg.tnln/1000));                              % tone length in samples
cfg.t           = (1:cfg.l) ./ cfg.Fs;                                          % time vector for individual tones
cfg.W           = tukeywin(cfg.l, 20/cfg.tnln)';                                % 10 ms on- and off-ramps 
if length(cfg.W) ~= length(cfg.t)
    if length(cfg.W) > length(cfg.t)
        cfg.W(round(end/2)) = [];
    end
end
cfg.SOA         = 0.5;                                                          % 500 ms target-tone SOA, also defines window size
cfg.n_wndws     = 10;                                                           % how many windows?
cfg.L           = floor(cfg.Fs * cfg.SOA * cfg.n_wndws);                        % overall trial stimulus length, in samples
cfg.F           = round(logspace(log10(239), log10(2045), 13));                 % log frequency space between 239 and 2924 Hz

cfg.n_mskrs     = 5;    %5 7  # of possible maskers within a given time segment   


% 
cfg.prtctdrgn   = 2;  %2                                                        % # of bands on either side of target in spectral protected region
cfg.n_trgfrqs   = 3;                                                            % # of possible target frequencies (489, 699, and 1000 Hz)
cfg.spacing     = [5; 2];                                                       % potential target frequencies: 
cfg.n_blocks    = 2;                                                            % # of blocks
cfg.n_trls      = 96;                                                          % # of trials / block
cfg.targ_p      = 0.75;                                                          % fraction of trials with target

% initialize target-tone freq vector over trials
cfg.F_targ = NaN(1, ceil((1-cfg.targ_p)*cfg.n_trls));
for i = 1:cfg.n_trgfrqs
	addTones = ones(1, cfg.n_trls*cfg.targ_p/cfg.n_trgfrqs)*cfg.F(cfg.spacing(1)+(i-1)*cfg.spacing(2));
    cfg.F_targ = [cfg.F_targ addTones];
end
cfg.trg_frqs = unique(cfg.F_targ(~isnan(cfg.F_targ))); % possible targ freqs

% initialize trigger codes in powers of 2 (for binary trigger words)
cfg.code = 7 * ones(1, cfg.n_trls);
for i = 1:cfg.n_trgfrqs
    cfg.code(cfg.F_targ==cfg.trg_frqs(i)) = 2^(i-1);
end

cfg.filename = cell(1, cfg.n_trls);

% loop over blocks
for block = 1:cfg.n_blocks
    
    fid = fopen(['Masker_Block_' num2str(block) '.txt'], 'w');
    
    % randomize block
    indices = randperm(cfg.n_trls);
    cfg.F_targ = cfg.F_targ(indices);
    cfg.code = cfg.code(indices);
    
    for trial = 1:cfg.n_trls

        strial = zeros(1, cfg.L);

        % Get masker-tone distribution based on target presence/absence
        F = cfg.F;
        if ~isnan(cfg.F_targ(trial))
            idx = find(cfg.F==cfg.F_targ(trial));
            % set the protected region by excluding adjacent two masker 
            % tones from target
            indcs = ones(1, length(cfg.F));
            indcs(idx-cfg.prtctdrgn:idx+cfg.prtctdrgn) = 0;            
            indcs = find(indcs);
            F = F(indcs);
        end
        
        for wndw = 1:cfg.n_wndws
            f = F(randperm(length(F))); % randomize masker freqs
            f = f(1:cfg.n_mskrs);
            
            ERB = 24.7*(4.37*(f/1000) +1);
            n = [[f - ERB/2]' [f + ERB/2]'];
            for ii = 1:length(f)
                f(ii) = round(unifrnd(n(ii,1),n(ii,2)));
            end
            
            if ~isnan(cfg.F_targ(trial)) % replace one masker with target
                if wndw > 1
                    f(1) = cfg.F_targ(trial);
                end

            end
            
            swin = build_window(f,cfg); % create current window of target + masker tones
            
            % add current segment to trial
            strt = (wndw-1) * round(cfg.SOA*cfg.Fs) + 1;
            fnsh = strt + round(cfg.SOA*cfg.Fs) - 1;
            strial(strt:fnsh) = swin;
        end
        
        cfg = write_trial(block, trial, strial, cfg, fid);   % create wav file
    end
    fclose(fid);
    save(['Masker_Block' num2str(block) '.mat'], 'cfg');    
end


function s = build_window(f,cfg)

    s = zeros(1, round(cfg.SOA*cfg.Fs));
    for i = 1:length(f)
        % jitter masker tone onset from 0-550 ms (target SOA - tone length)
        % jitter masker tone freq within an ERB
        if i ~= 1
            shft = (cfg.SOA - cfg.l/cfg.Fs) * rand(1); 
            erb = 24.7 * (0.00437*f(i) + 1);
            f_new = f(i) - erb + rand(1)*2*erb;
        else
            shft = 0;
            f_new = f(i);
        end        
        strt = round(shft*cfg.Fs) + 1;
        fnsh = strt + cfg.l - 1;
        tn = sin(2*pi*f_new*cfg.t);
        tn = tn .* (1/cfg.n_mskrs) .* cfg.W; % normalize and window
        s(strt:fnsh) = s(strt:fnsh) + tn;    
    end

end


function cfg = write_trial(block, trial, strial, cfg, fid)
    savedir = fullfile(pwd, ['Masker_Block_', num2str(block)]);
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
