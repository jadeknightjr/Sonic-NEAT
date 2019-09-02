import retro

env = retro.make("SonicTheHedgehog-Genesis", "GreenHillZone.Act1")
#env = retro.make("GoldenAxe-Genesis", "1Player.Arcade.DefaultCharacter.Level1")

env.reset()

done = False

while not done:     # done is die 3 times by default
    env.render()    # Displays sonic

    action = [0,0,1,0,0,0,0,1,1,1,0,0]
    #action = env.action_space.sample()     # Does random button presses
    #print(action) # The 1/0s tell you whether or not a button on genesis is on/off

    ob, rew, done, info = env.step(action)       # ob = image of screen at time of action
    print("Action ", action, "Reward ", rew)     # rew = amount of reward that he earned from whatever in scenario file
                                                 # done = whether done condition met
                                                 # info = dict of all values set in data.json