import random, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import BOARD_SIZE, PIECE_SIZE, ROTATIONS, LR_MOVES
from mechanics.tetris import TetrisGame
from experience_replayer import ExperienceReplayer

# Computes the probabilities of taking the next action, given a board state and piece
class ActionProbabilities(nn.Module):
    def __init__(self):
        super(ActionProbabilities, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(BOARD_SIZE + PIECE_SIZE, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            # nn.Linear(64, 64),
            # nn.GELU(),
            nn.Linear(64, ROTATIONS + LR_MOVES),
        )

    # Returns probabilities for rotations & left-right moves
    def forward(self, board, piece):
        x = torch.cat((board, piece), dim=0)  # Concatenate the flattened inputs
        x = self.nn(x)

        rotation_probs = F.softmax(x[:ROTATIONS], dim=0)
        lr_steps_probs = F.softmax(x[ROTATIONS:], dim=0)
        return (rotation_probs, lr_steps_probs)

class RLPlayer:
    def __init__(self, greedy_prob=0.2):
        self.model = ActionProbabilities()
        self.greedy_prob = greedy_prob

    def choose_action(self, game):
        # Convert the board and piece numpy arrays into tensors, and flatten them into 1 dimension
        board = torch.from_numpy(game.board).float().reshape(-1)
        piece = torch.from_numpy(game.current_piece.shape).float().reshape(-1)

        rotation_probs, steps_probs = self.model(board, piece)

        if random.random() < self.greedy_prob:
            # Choose the highest probability action
            rotations = torch.argmax(rotation_probs)
            lr_steps = torch.argmax(steps_probs) - ROTATIONS - 1
        else:
            # Sample from the probability distribution
            rotations = torch.distributions.Categorical(probs=rotation_probs).sample().item()
            lr_steps = torch.distributions.Categorical(probs=steps_probs).sample().item() - ROTATIONS - 1

        # print(f"RL player chose to rotate {rotations} time(s), and moved {lr_steps} steps")
        return rotations, lr_steps


class TrainRLPlayer:
    @staticmethod
    def load(player, filepath):
        player.load_state_dict(torch.load(filepath, weights_only=True))
        player.eval()

    @staticmethod
    def save(player, filepath):
        torch.save(player.state_dict(), filepath)

    @staticmethod
    def train(player, epochs, batch_size, lr):
        optimizer = torch.optim.Adam(player.model.parameters(), lr=lr)

        for epoch in range(epochs):
            # Zero the gradients
            optimizer.zero_grad()

            experiences = ExperienceReplayer.play(player, batch_size)
            loss = 0
            for experience in experiences:
                (state, action, reward) = experience
                (cur_board, cur_piece) = state
                (rotations, lr_steps) = action

                # Offset lr_steps to array indices
                lr_steps = lr_steps + ROTATIONS + 1

                # Get computed probabilities from current model
                board = torch.from_numpy(cur_board).float().reshape(-1)
                piece = torch.from_numpy(cur_piece.shape.copy()).float().reshape(-1)
                rotation_probs, steps_probs = player.model(board, piece)

                # if reward > 0:
                #     print("Reward:", reward)

                # TODO: Should I be using log probabilities?
                # loss += -torch.log(rotation_probs[rotations]) * reward
                # loss += -torch.log(steps_probs[lr_steps]) * reward

                # TODO: Use discounted rewards instead
                loss += -rotation_probs[rotations] * reward
                loss += -steps_probs[lr_steps] * reward
            
            # Backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
