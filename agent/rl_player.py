import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from constants import BOARD_SIZE, PIECE_SIZE, ROTATIONS, LR_MOVES

# Computes the probabilities of taking the next action, given a board state and piece
class ActionProbabilities(nn.Module):
    def __init__(self):
        super(ActionProbabilities, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(BOARD_SIZE + PIECE_SIZE, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, ROTATIONS + LR_MOVES),
        )

    # Returns unnormalized probabilities
    def forward(self, board, piece):
        x = torch.cat((board, piece), dim=0)  # Concatenate the flattened inputs
        x = self.nn(x)
        return x

class RLPlayer:
    def __init__(self, greedy_prob=0.2):
        self.probs = ActionProbabilities()
        self.greedy_prob = greedy_prob

    def choose_action(self, game):
        # Convert the board and piece numpy arrays into tensors, and flatten them into 1 dimension
        board = torch.from_numpy(game.board).float().reshape(-1)
        piece = torch.from_numpy(game.current_piece.shape).float().reshape(-1)

        logits = self.probs(board, piece)
        rotation_probs = F.softmax(logits[:ROTATIONS], dim=0)
        steps_probs = F.softmax(logits[ROTATIONS:], dim=0)

        if random.random() < self.greedy_prob:
            # Choose the highest probability action
            rotations = torch.argmax(rotation_probs)
            lr_steps = torch.argmax(steps_probs) - ROTATIONS - 1
        else:
            # Sample from the probability distribution
            rotations = torch.distributions.Categorical(probs=rotation_probs).sample().item()
            lr_steps = torch.distributions.Categorical(probs=steps_probs).sample().item() - ROTATIONS - 1

        print(f"RL player chose to rotate {rotations} time(s), and moved {lr_steps} steps")
        return rotations, lr_steps

