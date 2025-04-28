import random, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import BOARD_SIZE, PIECE_SIZE, ROTATIONS, LR_MOVES, LR_OFFSET
from mechanics.tetris import TetrisGame, apply_moves
from experience_replayer import ExperienceReplayer

class BoardStateValue(nn.Module):
    def __init__(self):
        super(BoardStateValue, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(BOARD_SIZE, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    # Returns probabilities for rotations & left-right moves
    def forward(self, board):
        x = self.nn(board)
        return x


class DQN_RLPlayer:
    def __init__(self, explore_prob=0.2):
        self.model = BoardStateValue()
        self.explore_prob = explore_prob

    def choose_action(self, game: TetrisGame):
        piece = game.current_piece

        if random.random() < self.explore_prob:  # Choose a random action
            action_idx = random.randrange(ROTATIONS * LR_MOVES)
            rotations = int(action_idx / LR_MOVES)
            lr_steps = (action_idx % LR_MOVES) - LR_OFFSET
            return rotations, lr_steps

        # Iterate through every combination of rotations & left-right steps
        # Compute the state value of applying the above move, and choose the state
        # with the highest value
        # qvalues, highest_qv = [], -torch.inf
        chosen = None
        cur_score = game.score
        highest_qv = -torch.inf
        for rotations in range(ROTATIONS):
            for lr_steps in range(LR_MOVES):
                lr_steps -= LR_OFFSET
                game_copy = copy.deepcopy(game)

                apply_moves(game_copy, rotations=rotations, lr_steps=lr_steps)
                score_diff = game_copy.score - cur_score

                new_board = torch.from_numpy(game_copy.board).float().reshape(-1)
                qvalue = self.model(new_board)

                total_value = qvalue + score_diff
                if  total_value > highest_qv:
                    chosen = (rotations, lr_steps)
                    highest_qv = total_value

        return chosen


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
        avg_score = 0

        for epoch in range(epochs):
            # Play the game {batch_size} times
            experiences = ExperienceReplayer.play(player, batch_size)

            # We want to compute discounted total rewards as the state value
            training_data = []
            total_score = 0
            for experience in experiences:
                # Each experience contains the whole play for 1 game
                # We will iterate the samples in reverse, so that earlier states can
                # include discounted future rewards to compute the "actual state value"
                discounted_future_reward = 0
                for sample in reversed(experience):
                    (_, _, reward, new_board) = sample
                    (score_diff, penalties) = reward

                    actual_svalue = discounted_future_reward - penalties
                    discounted_future_reward = discount * (discounted_future_reward + score_diff)

                    training_data.append((new_board, actual_svalue))
                    total_score += score_diff


            # Shuffle the training data and actually train the model
            # TODO: Should be I creating a separate target network?
            random.shuffle(training_data)
            epoch_loss = 0

            for data in training_data:
                optimizer.zero_grad()

                (new_board, actual_svalue) = data

                new_board = torch.from_numpy(new_board).float().reshape(-1)
                model_svalue = player.model(new_board)

                # TODO: Should I be computing loss over a batch?
                loss = (actual_svalue - model_svalue) ** 2
                loss.backward()
                optimizer.step()

                epoch_loss += loss

            # if (epoch + 1) % 100 == 0:
            #     print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
            print(f'Epoch {epoch+1}, Epoch loss: {epoch_loss.item():.0f}, Avg score: {(total_score / len(training_data)):.2f}')
