# Sonic-NEAT
This is the repository of me experimenting with RetroAI to make Sonic the Hedgehog beat the game.

## playback.py
This simply runs the .pkl that solved GreenHillZone.Act1

## tut1.py
Essentially this is what happens when every movement is random and no learning is really taking place

## tut2.py
This is the version of NEAT that is unorganized and doesn't actually solve the game. It runs until a counter hits 250(basically stuck in a generation). No endstate as the while loop never ends.

## tut3.py
The upgraded version of tut2.py that organizes everything into a class. Actually can solve the game given enough generations

## tut3_alt.py
This is tut3.py but allows you to see what the AI actually sees pixelwise.
