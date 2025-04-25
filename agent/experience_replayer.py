import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechanics.tetris import TetrisGame, apply_moves

class ExperienceReplayer:

    # Runs the game multiple times, and returns the replay experiences
    # Each experience will be a full iteration of the game, and store
    # an array of (state, action, reward) tuples, where:
    #   state = (current_board, piece)
    #   action = (rotations, lr_steps)
    #   reward = change in score after the action completes
    @staticmethod
    def play(player, num_times=10):
        experiences = []
        for _ in range(num_times):
            game = TetrisGame()
            samples = []

            while not game.game_over:
                cur_board = game.board
                cur_piece = game.current_piece
                cur_score = game.score

                # Ask the agent to compute actions, and apply them
                rotations, lr_steps = player.choose_action(game)
                apply_moves(game, rotations, lr_steps)
                score_delta = game.score - cur_score

                # Spawn a new piece if needed
                game.spawn_piece()

                # Two boards can have the same score, so we want to increase the value
                # of the terminal state by how much the board is filled.
                # Further, we give a higher value to lower rows, to encourage higher
                # fill density.
                # We scale this value down since we don't want it to dominate over finishing rows.
                if game.game_over:
                    fill_score = 0
                    for i, row in enumerate(game.board):
                        fill_score += i * np.count_nonzero(row)
                    fill_score /= 1000.0
                    score_delta += fill_score

                # Collect an experience - (state, action, reward)
                sample = ((cur_board, cur_piece), (rotations, lr_steps), score_delta / 10)
                samples.append(sample)
            
            # Collect the samples from a whole iteration
            experiences.append(samples)

        return experiences
