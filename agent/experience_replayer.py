import sys, os
import numpy as np
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechanics.tetris import TetrisGame, apply_moves

class ExperienceReplayer:

    @staticmethod
    def compute_penalties(board):
        penalties = 0

        # Penalize for holes in rows
        # Calculate holes: empty cells below filled cells in the same column
        holes = 0
        # Iterate through each column
        for col in range(board.shape[1]):
            found_block = False
            # Scan down the column from top to bottom
            for row in range(board.shape[0]):
                if board[row, col] != 0:
                    # We found the first block in this column
                    found_block = True
                elif found_block:
                    # This cell is empty, but we've already seen a block above it in this column
                    # Therefore, it's a hole.
                    holes += 1
        
        # Apply a penalty based on the number of holes found.
        hole_penalty_factor = 0.25
        penalties += holes * hole_penalty_factor

        # Penalize for bumpiness
        # Calculate column heights
        heights = [0] * board.shape[1]
        for col in range(board.shape[1]):
            for row in range(board.shape[0]):
                if board[row, col] != 0:
                    # Height is the number of rows from the bottom up to the highest block
                    heights[col] = board.shape[0] - row
                    break # Found the highest block in this column

        # Calculate bumpiness (square of differences between adjacent column heights)
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += (heights[i] - heights[i+1]) ** 2

        # Apply a penalty based on the bumpiness.
        bumpiness_penalty_factor = 0.1
        penalties += bumpiness * bumpiness_penalty_factor

        # Two boards can have the same score, so we want to increase the value
        # of the terminal state by how much the board is filled.
        # Further, we give a higher value to lower rows, to encourage higher
        # fill density.
        # We scale this value down since we don't want it to dominate over finishing rows.
        # if game.game_over:
        #     fill_score = 0
        #     for i, row in enumerate(board):
        #         fill_score += i * np.count_nonzero(row)
        #     fill_score /= 100.0
        #     reward += fill_score

        return penalties

    # Runs the game multiple times, and returns the replay experiences
    # Each experience will be a full iteration of the game, and store
    # an array of (state, action, reward, new_state) tuples, where:
    #   state = (current_board, piece)
    #   action = (rotations, lr_steps)
    #   reward = (score_change, penalties)
    #   new_state = new_board_position
    @staticmethod
    def play(player, num_times=10):
        experiences = []
        for _ in range(num_times):
            game = TetrisGame()
            samples = []

            while not game.game_over:
                # TODO: Can these be shallow copy instead?
                cur_board = copy.deepcopy(game.board)
                cur_piece = copy.deepcopy(game.current_piece)
                prev_score = game.score

                # Ask the agent to compute actions, and apply them
                rotations, lr_steps = player.choose_action(game)
                apply_moves(game, rotations, lr_steps)
                # Spawn a new piece if needed
                game.spawn_piece()

                penalties = ExperienceReplayer.compute_penalties(game.board)
                reward = (game.score - prev_score, penalties)

                new_state = copy.deepcopy(game.board)

                # Collect a sample - (state, action, reward, new_state)
                sample = ((cur_board, cur_piece), (rotations, lr_steps), reward, new_state)
                samples.append(sample)
            
            # Collect the samples from a whole iteration
            experiences.append(samples)

        return experiences
