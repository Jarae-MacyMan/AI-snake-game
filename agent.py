import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0 #number of games 
        self.epsilon = 0 #controll randomness
        self.gamma = 0.9 #discount rate < 1
        self.memory = deque(maxlen=MAX_MEMORY) #popleft() if it exseeds memory limit
        self.model = Linear_QNet(11, 256, 3) #11 states output 3 hidden input and ouput 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
       

    def get_state(self, game):
        head = game.snake[0]
        #4 points around the head
        point_l = Point(head.x -20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y -20)
        point_d = Point(head.x, head.y +20, )

        #booleans checking direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        #list of 11 game states Danger = direction of the sides relative to sanke head 
        state = [  #[ danger 0,0,0  direction 0,0,0,0,   food location 0,0,0,0]
            #Danger straight dependent on curr direction
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)), 

            #Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
            
            #Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            #Move direction
            dir_l, 
            dir_r,
            dir_u,
            dir_d,

            #Food loacation
            game.food.x < game.head.x, #food left
            game.food.x > game.head.x, #food right
            game.food.y < game.head.y, #food up
            game.food.y > game.head.y #food down
        ]

        return np.array(state, dtype=int) #convert list to numpy arr and convert values to an int of 1 qnd 0


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #pop left if max memory is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else: 
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #random moves: trade off btw exploration vs exploitation
        self.epsilon = 80 - self.n_games #based on number of games so far more games the smaller the epsilon
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon: #smaller the epsilon = less random
            move = random.randint(0, 2)
            final_move[move] = 1 #puts a 1 in a random place in the final move
        else: #move based on model not random
            state0 = torch.tensor(state, dtype=torch.float) #convert to tensor
            prediction = self.model(state0) #predict action based off state
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
            

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True: 
        #get old state 
        state_old = agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old) #action based on the state

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember 
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #train the long memory exp replay trains on all prev moves plot result
            game.reset()
            agent.n_games += 1 #inc number of games
            agent.train_long_memory()

            if score > record: #if we have a new high score 
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ ==  '__main__':
    train()

