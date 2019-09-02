# Sonic-NEAT
This is the repository of me experimenting with RetroAI to make Sonic the Hedgehog beat the game.

## playback.py
This simply runs the .pkl that solved GreenHillZone.Act1 . This file also creates a .bk2 file from the .pkl file that you are running(in this case, it reads in the only included .pkl file). RetroAI includes a playback_movie.py which allows you to convert that .bk2 file into any format(I chose .mp4).
```
# Example
python3 -m retro.scripts.playback_movie SonicTheHedgehog-Genesis-GreenHillZone.Act1-000000.bk2
```

## tut1.py
Essentially this is what happens when every movement is random and no learning is really taking place


### Environment
This is how our environment is setup in RetroAI.
```
# Load in the Game and Stage you Want
# env = retro.make(GAME, STAGE)
env = retro.make("SonicTheHedgehog-Genesis", "GreenHillZone.Act1")
# You will want to run env.reset() before you do anything
env.reset()
```

### Action
In order to make our AI do something in our environment, we feed it an Action
```
action = [0,0,1,0,0,0,0,1,1,1,0,0].     # Each index represents a different key/button on a Genesis controller
#action = env.action_space.sample()     # Does random button presses
#print(action)                          # The 1/0s tell you whether or not a button on genesis is on/off
```
### Results
In each iteration, we may want to update information resulting from using the said action. Printing something out at each iteration can be helpful to see how our AI is doing.
``` 
    ob, rew, done, info = env.step(action)       # ob = image of screen at time of action
    print("Action ", action, "Reward ", rew)     # rew = amount of reward that he earned from whatever in scenario file
                                                 # done = whether done condition met
                                                 # info = dict of all values set in data.json
```
### BONUS: See Sonic
This doesn't help the AI run, but it does show what your AI is currently doing. Really fun to watch...
```    
env.render()    # Displays sonic
```
## tut2.py
This is the version of NEAT that is unorganized and doesn't actually solve the game. It runs until a counter hits 250(basically stuck in a generation). No endstate as the while loop never ends.

## tut3.py
The upgraded version of tut2.py that organizes everything into a class. Actually can solve the game given enough generations

## tut3_alt.py
This is tut3.py but allows you to see what the AI actually sees pixelwise.
