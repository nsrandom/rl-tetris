import torch

from constants import LR_MOVES, LR_OFFSET
from experience_replayer import ExperienceReplayer

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
                    (state, action, reward) = sample
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