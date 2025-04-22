import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechanics.tetris import TetrisGame

class ExperienceReplayer:

    @staticmethod
    def apply_moves(game, rotations, lr_steps):
        # First rotate
        for _ in range(rotations):
            game.rotate_piece()
        
        # Then move left-right 1 step at a time
        # This prevents the move from being rejected if the agent
        # computes more steps than what's valid
        if lr_steps != 0:
            direction = 1 if lr_steps > 0 else -1
            num_steps = abs(lr_steps)
            for _ in range(num_steps):
                game.move_piece(0, direction)
        
        # Finally, hard drop
        while game.move_piece(1, 0):
            pass

    # Runs the game multiple times, and returns the replay experiences
    # Each experience will be a (state, action, reward) tuple, where:
    #   state = (current_board, piece)
    #   action = (rotations, lr_steps)
    #   reward = change in score after the action completes
    @staticmethod
    def play(player, num_times=10):
        experiences = []
        for _ in range(num_times):
            game = TetrisGame()

            while not game.game_over:
                cur_board = game.board
                cur_piece = game.current_piece
                cur_score = game.score

                # Ask the agent to compute actions, and apply them
                rotations, lr_steps = player.choose_action(game)
                ExperienceReplayer.apply_moves(game, rotations, lr_steps)
                score_delta = game.score - cur_score

                # Spawn a new piece if needed
                game.spawn_piece()

                # Two boards can have the same score, so we want to increase the value
                # of the terminal state by how much the board is filled.
                if game.game_over:
                    score_delta += np.count_nonzero(game.board) / 10.0

                # Collect an experience - (state, action, reward)
                experience = ((cur_board, cur_piece), (rotations, lr_steps), score_delta)
                experiences.append(experience)

        return experiences
