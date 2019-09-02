import retro
import numpy as np
import cv2
import neat
import pickle

env = retro.make("SonicTheHedgehog-Genesis", "GreenHillZone.Act2")
# inx, iny, inc = env.observation_space.shape # size of image created by emulator (resolution of screen) (x,y,#colors)
# print(inx,iny,inc)

imgArray = []
xpos_end = 0

def eval_genomes(genomes, config):

    for genome_id, genome in genomes:   # Currently 20 (pop_size in config) neuralnets
        ob = env.reset()                # Observation variable(image)... will become input of neuralnet
        ac = env.action_space.sample()  # action variable... will be generic sample

        inx, iny, inc = env.observation_space.shape # size of image created by emulator (resolution of screen) (x,y,#colors)
        inx = int(inx/8)
        iny = int(iny/8)
        
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0     # output from neuralnet of how sucessful(from reward fn)
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        done = False
        # cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        
        while not done:

            env.render()

            frame += 1
            # To see what the network actually sees
            # scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            # scaledimg = cv2.resize(scaledimg, (iny, inx))

            ob = cv2.resize(ob, (inx,iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)    # Make gray-scale (v inputs, ^ faster)
            ob = np.reshape(ob, (inx,iny))

            # cv2.imshow('main', scaledimg)
            # cv2.waitKey(1)

            imgArray = list(ob.flatten())
            # for x in ob:
            #     for y in ob:
            #         imgArray.append(y)

            nnOutput = net.activate(imgArray)

            # print(len(imgArray), nnOutput)
            ob, rew, done, info = env.step(nnOutput) # increments emulator state by one
            imgArray.clear() 

            # xpos = info['x']
            # xpos_end = info['screen_x_end']
            # if xpos > xpos_max:
            #     fitness_current += 1
            #     xpos_max = xpos

            # if xpos == xpos_end and xpos > 500:
            #     fitness_current += 100000
            #     done = True

            fitness_current += rew
            print("Current Max: ", current_max_fitness, "Current Fit: ", fitness_current)

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
            
            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)
            
            genome.fitness = fitness_current
             



config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
    
