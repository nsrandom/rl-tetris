import random, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import BOARD_SIZE, PIECE_SIZE, ROTATIONS, LR_MOVES, LR_OFFSET
from mechanics.tetris import TetrisGame
from experience_replayer import ExperienceReplayer

# Computes the probabilities of taking the next action, given a board state and piece
class ActionProbabilities(nn.Module):
    def __init__(self):
        super(ActionProbabilities, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(BOARD_SIZE + PIECE_SIZE, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            # Each combination of rotations and left/right steps is a distinct action
            nn.Linear(256, ROTATIONS * LR_MOVES),
        )

    # Returns probabilities for rotations & left-right moves
    def forward(self, board, piece):
        x = torch.cat((board, piece), dim=0)  # Concatenate the flattened inputs
        x = self.nn(x)

        x = F.softmax(x, dim=0)
        return x


class PolicyGradient_RLPlayer:
    def __init__(self, explore_prob=0.2):
        self.model = ActionProbabilities()
        self.explore_prob = explore_prob

    def choose_action(self, game):
        # Convert the board and piece numpy arrays into tensors, and flatten them into 1 dimension
        board = torch.from_numpy(game.board).float().reshape(-1)
        piece = torch.from_numpy(game.current_piece.shape).float().reshape(-1)

        # Compute the action probabilities from the NN model
        action_probs = self.model(board, piece)

        if random.random() < self.explore_prob:  # Choose a random action
            action_idx = random.randrange(ROTATIONS * LR_MOVES)
        else:  # Sample from the probability distribution
            action_idx = torch.distributions.Categorical(probs=action_probs).sample().item()

        rotations = int(action_idx / LR_MOVES)
        lr_steps = (action_idx % LR_MOVES) - LR_OFFSET
        return rotations, lr_steps

class TrainRLPlayer:
    @staticmethod
    def load(player, filepath):
        player.model.load_state_dict(torch.load(filepath, weights_only=True))
        player.model.eval()

    @staticmethod
    def save(player, filepath):
        torch.save(player.model.state_dict(), filepath)

    @staticmethod
    def train(player, epochs, batch_size, lr, discount=0.99):
        optimizer = torch.optim.Adam(player.model.parameters(), lr=lr)

        for epoch in range(epochs):
            # Zero the gradients
            optimizer.zero_grad()

            experiences = ExperienceReplayer.play(player, batch_size)
            loss = 0
            for experience in experiences:
                # Each experience contains the whole play for 1 game
                # We will iterate the samples in reverse, so that earlier
                # states can include discounted future rewards
                discounted_future_reward = 0
                for sample in reversed(experience):
                    (state, action, reward, _) = sample
                    (cur_board, cur_piece) = state
                    (rotations, lr_steps) = action

                    # Offset lr_steps to array indices
                    lr_steps = lr_steps + LR_OFFSET

                    # Get computed probabilities from current model
                    board = torch.from_numpy(cur_board).float().reshape(-1)
                    piece = torch.from_numpy(cur_piece.shape.copy()).float().reshape(-1)
                    action_probs = player.model(board, piece)

                    total_reward = reward + discounted_future_reward
                    # Probabilities are (0,1), hence log(prob) < 0
                    # We want to maximize reward, hence we will define loss as negative reward, and minimize loss
                    loss += -torch.log(action_probs[rotations * LR_MOVES + lr_steps]) * total_reward

                    # Update discounted reward for next iteration
                    discounted_future_reward = discount * total_reward
            
            # Backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
            # if (epoch + 1) % 100 == 0:
            #     print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
