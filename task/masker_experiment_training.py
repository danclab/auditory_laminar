import math
import os
import pandas as pd

from psychopy import sound, core, data, clock, parallel, gui, visual, event
from psychopy.hardware import keyboard
import psychtoolbox as ptb

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.2.5'
expName = 'masker'  # from the Builder filename that created this script
expInfo = {
    'participant': '',
    'session': '',
}
# --- Show participant info dialog --
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s_training.csv' % (expInfo['participant'], expName, expInfo['date'])

# Data to write to log file
columns = [
    "id",
    "session",
    "block",
    "trial",
    "stim_file",
    "sound_start",
    "detection",
    "sound_stop",
    "button_press",
]
data_dict = {i: [] for i in columns}

# --- Setup the Window ---
win = visual.Window(
    size=[1920, 1080], fullscr=True, screen=0,
    winType='pyglet', allowStencil=False,
    monitor='testMonitor', color=[0, 0, 0], colorSpace='rgb',
    blendMode='avg', useFBO=True,
    units='height')
win.mouseVisible = False
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 120.0  # could not measure, so guess

defaultKeyboard = keyboard.Keyboard(backend='iohub')

# Create parallel port interface for sending events to MEG
addressPortParallel = '0x3FE8'
port = parallel.ParallelPort(address=addressPortParallel)
# Initialize to 0
port.setData(0)

# Event triggers
#trig_start = 10
trig_stop = 20
trig_trial_keypress = 70
trig_start_keypress = 100


# Send event trigger
def trigger(send_bit):
    port.setData(send_bit)
    core.wait(0.004)
    port.setData(0)
    
# --- Initialize components for Routine "instructions_start" ---
start_text = visual.TextStim(win=win, name='start_text',
                             text='Welcome\n\nPress a button to continue.',
                             font='Open Sans',
                             pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0,
                             color='white', colorSpace='rgb', opacity=None,
                             languageStyle='LTR',
                             depth=0.0)

# --- Initialize components for Routine "instructions_block" ---
block_instructions = visual.TextStim(win=win, name='block_instructions',
                                     text='While the sound is playing, press a button as soon as you start to hear out the repeating tones from the background.\n\n\nYou will have to press a button to start each sound. If you get tired, you may also wait a short moment before starting the next sound.\n\nRemember to sit still and keep your eyes fixed on the + at the center of the screen.\n\n\nWait for the experimenter to start the experiment.',
                                     font='Open Sans',
                                     pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0,
                                     color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
                                     languageStyle='LTR',
                                     depth=0.0)
block_keypress = keyboard.Keyboard()

# --- Initialize components for Routine "trial" ---
stimulus = sound.backend_ptb.SoundPTB('A', secs=5,
                                      stereo=True, hamming=False, name='stimulus', syncToWin=None, blockSize=64,
                                      sampleRate=48000)

fixation = visual.TextStim(win=win, name='fixation',
                           text='+',
                           font='Open Sans',
                           pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0,
                           color='white', colorSpace='rgb', opacity=None,
                           languageStyle='LTR',
                           depth=-1.0)

# --- Initialize components for Routine "post_trial" ---
post_trial_fixation = visual.TextStim(win=win, name='post_trial_fixation',
                                      text='+',
                                      font='Open Sans',
                                      pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0,
                                      color='white', colorSpace='rgb', opacity=None,
                                      languageStyle='LTR',
                                      depth=0.0)

# --- Initialize components for Routine "instructions_final" ---
final_text = visual.TextStim(win=win, name='final_text',
                             text='Thank you\n\nPress the "escape" key to exit.',
                             font='Open Sans',
                             pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0,
                             color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
                             languageStyle='LTR',
                             depth=0.0)
final_keypress = keyboard.Keyboard()

# Use the global clock for the log file
globalClock = core.Clock()

# Show the start text until the participant presses a button
start_text.setAutoDraw(True)
win.flip()
event.waitKeys(keyList=['space'])
start_text.setAutoDraw(False)

# Load block conditions
blocks = data.TrialHandler(nReps=1.0, method='sequential',
                           extraInfo=expInfo, originPath=-1,
                           trialList=data.importConditions('masker_conditions.xlsx'),
                           seed=None, name='blocks')
thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values

exit_exp = False

for block_idx, thisBlock in enumerate(blocks):
    if thisBlock != None:
        for paramName in thisBlock:
            exec('{} = thisBlock[paramName]'.format(paramName))

    # Show block instructions until the experimenter presses the space bar
    block_instructions.setAutoDraw(True)
    win.flip()
    event.waitKeys(keyList=['space'])
    block_instructions.setAutoDraw(False)
    
    # Start MEG recording
    trigger(252)

    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='sequential',
                               extraInfo=expInfo, originPath=-1,
                               trialList=data.importConditions(masker_condition_file),
                               seed=None, name='trials')

    # Show fixation and wait 100ms
    fixation.setAutoDraw(True)
    win.flip()
    core.wait(.1)

    for trial_idx, thisTrial in enumerate(trials):
        print('Trial {}'.format(trial_idx+1))
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                exec('{} = thisTrial[paramName]'.format(paramName))

        # Set sound stimulus
        stimulus.setSound(sound_in, secs=5, hamming=False)
        stimulus.setVolume(2.0, log=False)

        # For some reason, the delay is only consistent when we use ptb.GetSecs() rather than the psychopy global clock
        # This schedules the sound to play 500ms from now
        delay = .5
        start_pbt_t = ptb.GetSecs()
        end_pbt_t = start_pbt_t + 5 + delay
        stimulus.play(when=start_pbt_t + delay, loops=None, log=False)
        # Wait until the sound is scheduled to play
        while ptb.GetSecs() < start_pbt_t + delay:
            pass
        # Send sound start trigger
        trigger(id)
        # Start time for log
        start_t = globalClock.getTime()
        # Time the tone was detected
        detect_t = float('NaN')
        # Look for button press until end of the sound
        while ptb.GetSecs() < end_pbt_t:
            if math.isnan(detect_t):
                key = event.getKeys()
                if 'space' in key:
                    trigger(trig_trial_keypress)
                    detect_t = globalClock.getTime()
        # Stop sound, send trigger, and get end time for log
        stimulus.stop(reset=True, log=False)
        trigger(trig_stop)
        end_t = globalClock.getTime()

        # Wait for button press
        event.waitKeys(keyList=['space'])
        trigger(trig_start_keypress)
        button_t = globalClock.getTime()

        # Add info to log and save just in case
        data_dict["id"].append(expInfo['participant'])
        data_dict["session"].append(expInfo['session'])
        data_dict["block"].append(block_idx)
        data_dict["trial"].append(trial_idx)
        data_dict["stim_file"].append(sound_in)
        data_dict["sound_start"].append(start_t)
        data_dict["sound_stop"].append(end_t)
        data_dict["button_press"].append(button_t)
        data_dict["detection"].append(detect_t)
        df = pd.DataFrame(data_dict)
        df.to_csv(filename)

        if exit_exp or event.getKeys(keyList=['q'], timeStamped=False):
            exit_exp = True
            break

    fixation.setAutoDraw(False)

    if exit_exp or event.getKeys(keyList=['q'], timeStamped=False):
        break

    # Stop MEG recording
    trigger(250)
    
df = pd.DataFrame(data_dict)
df.to_csv(filename)
win.close()
core.quit()
