from os import stat
import torch
import random
import numpy as np
from collections import deque
from game import BLOCK_SIZE, SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

LR = 0.001
MAX_MEMORY = 100_000
BATCH_SIZE = 1000

class Agent: 
    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(14, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game): 
        dr = game.direction == Direction.RIGHT
        dl = game.direction == Direction.LEFT
        du = game.direction == Direction.UP
        dd = game.direction == Direction.DOWN
        BLOCK_SIZE = 20
        rp = Point(game.head.x + BLOCK_SIZE, game.head.y)
        lp = Point(game.head.x - BLOCK_SIZE, game.head.y)
        up = Point(game.head.x, game.head.y + BLOCK_SIZE)
        dp = Point(game.head.x, game.head.y - BLOCK_SIZE)
        idx = game.clockwise.index(game.direction)
        left_turn_dir = game.clockwise[(idx - 1) % len(game.clockwise)]
        right_turn_dir = game.clockwise[(idx + 1) % len(game.clockwise)]
        state = [
            # danger straight
            dr and game.is_collision(rp) or
            dl and game.is_collision(lp) or
            du and game.is_collision(up) or
            dd and game.is_collision(dp), 
            # danger right
            dr and game.is_collision(dp) or
            dl and game.is_collision(up) or
            du and game.is_collision(rp) or
            dd and game.is_collision(lp), 
            # danger left
            dr and game.is_collision(up) or
            dl and game.is_collision(dp) or
            du and game.is_collision(lp) or
            dd and game.is_collision(rp),

            # direction
            dl, dr, du, dd, 

            # food direction
            game.head.x > game.food.x, # left
            game.head.x < game.food.x, # right
            game.head.y > game.food.y, # down
            game.head.y < game.food.y, # up

            
            game.is_future_body_collision(game.head, left_turn_dir),
            game.is_future_body_collision(game.head, game.direction),
            game.is_future_body_collision(game.head, right_turn_dir)
        ]

        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action, reward, next_state, done))

    def train_short_term(self, state, action, reward, next_state, done): 
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_term(self): 
        if len(self.memory) > BATCH_SIZE: 
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else: 
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state): 
        self.epsilon = 80 - self.n_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon: 
            move = random.randint(0, 2)
            action[move] = 1
        else: 
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action


def train(): 
    plot_score = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True: 
        old_state = agent.get_state(game)
        action = agent.get_action(old_state)
        reward, done, score = game.play_step(action)
        new_state = agent.get_state(game)

        agent.train_short_term(old_state, action, reward, new_state, done)
        agent.remember(old_state, action, reward, new_state, done)
        
        if done: 
            print('old_state[-1] = ', old_state[-1])
            print('old_state[-2] = ', old_state[-2])
            print('old_state[-3] = ', old_state[-3])
            game.reset()
            agent.n_games += 1
            agent.train_long_term()
            if score > record: 
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', record)
            # append to score and mean_score and then plot
            plot_score.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_score, plot_mean_score)


if __name__ == '__main__': 
    train()