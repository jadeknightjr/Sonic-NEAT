import retro        # pip install gym-retro
import numpy as np  # pip install numpy
import cv2          # pip install opencv-python
import neat         # pip install neat-python
import pickle       # pip install cloudpickle


class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        
    def work(self):
        
        self.env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act2')
        
        self.env.reset()
        
        ob, _, _, _ = self.env.step(self.env.action_space.sample())

        inx, iny, inc = self.env.observation_space.shape 
        inx = int(inx/8)
        iny = int(iny/8)
        
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        fitness = 0
        counter = 0
        imgarray = []
        
        xpos = 0
        xpos_max = 0

        cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        done = False

        while not done:
            # self.env.render()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            
            cv2.imshow('main', ob)
            cv2.waitKey(1)

            imgarray = np.ndarray.flatten(ob)
            
            actions = net.activate(imgarray)
            
            ob, rew, done, info = self.env.step(actions)

            xpos = info['x']
            
            if xpos > xpos_max:
                xpos_max = xpos
                counter = 0
                fitness += 1
            else:
                counter += 1
                
            if counter > 250:
                done = True
                
            if xpos >= info['screen_x_end'] and xpos > 500:
                print("xpos: ", xpos, "screen_x_end", info['screen_x_end'])

                fitness += 100000
                done = True
                
        print("fitness: ", fitness)
        return fitness

def eval_genomes(genome, config):
    
    worky = Worker(genome, config)
    return worky.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

pe = neat.ParallelEvaluator(6, eval_genomes)

winner = p.run(pe.evaluate)

with open('[Act2]winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

