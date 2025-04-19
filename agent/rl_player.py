import numpy as np
import torch
import torch.nn as nn
import random

BOARD_SIZE = 20 * 10
PIECE_SIZE = 4 * 4
ROTATIONS = 4 # 0, 90, 180 and 270 degrees
LR_MOVES = 11 # -5 to +5

class QValue(nn.Module):
    def __init__(self):
        super(QValue, self).__init__()
        self.fc1 = nn.Linear(BOARD_SIZE + PIECE_SIZE, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, ROTATIONS + LR_MOVES)

    def forward(self, board, piece):
        x = torch.cat((board, piece), dim=0)  # Concatenate the flattened inputs
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class RLPlayer:
    def __init__(self):
        self.qvalue = QValue()
        self.greedy_prob = 0.8

    def choose_action(self, game):
        # Convert the board and piece numpy arrays into tensors, and flatten them into 1 dimension
        board = torch.from_numpy(game.board).float().reshape(-1)
        piece = torch.from_numpy(game.current_piece.shape).float().reshape(-1)

        logits = self.qvalue(board, piece)
        if random.random() < self.greedy_prob:
            rotations = torch.argmax(logits[:ROTATIONS])
            lr_steps = torch.argmax(logits[ROTATIONS:]) - ROTATIONS - 5
        else:
            rotations = random.randint(0, 3)
            lr_steps = random.randint(-4, 4)

        return rotations, lr_steps

