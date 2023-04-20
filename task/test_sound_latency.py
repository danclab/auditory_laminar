import os
import pandas as pd
from psychopy import prefs
prefs.hardware['audioLib'] = 'PTB'
prefs.hardware['audioDevice']='Haut-parleurs (Sound Blaster Audigy 5/Rx)'
prefs.hardware['audioLatencyMode']=4
from psychopy import sound, core, data, clock, parallel, gui, visual, event
from psychopy.hardware import keyboard
import psychtoolbox as ptb
from serial import Serial

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.2.5'
expName = 'oddball'  # from the Builder filename that created this script

# --- Setup the Window ---
win = visual.Window(
    size=[1920, 1080], fullscr=True, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
win.mouseVisible = False
# store frame rate of monitor if we can measure it
    
defaultKeyboard = keyboard.Keyboard(backend='iohub')
addressPortParallel = '0x3FE8'
port = parallel.ParallelPort(address=addressPortParallel)

trig_start=10
trig_stop=20

def trigger(send_bit):
    port.setData(send_bit)
    core.wait(0.004)
    port.setData(0)
trigger(0)

#globalClock = core.Clock()

# Start recording
#trigger(252)
    

stimulus = sound.backend_ptb.SoundPTB('stimuli/Oddball_Block_1/trial_03.wav', secs=5, 
    stereo=True, hamming=False, name='stimulus', syncToWin=None, blockSize=64,
    sampleRate=48000)
#stimulus.setVolume(2.0, log=False)

core.wait(.5)

for trial_idx in range(10):        
    #start_t=globalClock.getTime()
    start_t=ptb.GetSecs()
    delay=.05
    end_t=start_t+5+delay
    stimulus.play(when=start_t+delay, loops=None, log=False)
        
    while ptb.GetSecs()<start_t+delay:
        pass
    trigger(trig_start)
    while ptb.GetSecs()<end_t:
        pass
    stimulus.stop(reset=True, log=False)
    trigger(trig_stop)
        
    core.wait(.5)    
    
    if event.getKeys(keyList=['q'], timeStamped=False):
        break

#tigger(250)

win.close()
core.quit()
