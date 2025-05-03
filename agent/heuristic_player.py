import copy

from constants import ROTATIONS, LR_MOVES, LR_OFFSET
from experience_replayer import ExperienceReplayer
from mechanics.tetris import TetrisGame, apply_moves

class HeuristicPlayer:

    def choose_action(self, game: TetrisGame):
        # Iterate through every combination of rotations & left-right steps
        # For each, compute the heuristic penalty value of applying the above move,
        # and choose the state with the highest (score_diff - penalty)

        chosen = None
        cur_score = game.score
        highest_value = None

        for rotations in range(ROTATIONS):
            for lr_steps in range(LR_MOVES):
                lr_steps -= LR_OFFSET
                game_copy = copy.deepcopy(game)

                apply_moves(game_copy, rotations=rotations, lr_steps=lr_steps)
                score_diff = game_copy.score - cur_score
                penalties = ExperienceReplayer.compute_penalties(game_copy)

                total_value = score_diff - penalties
                if not highest_value:
                    highest_value = total_value
                    chosen = (rotations, lr_steps)
                if total_value > highest_value:
                    chosen = (rotations, lr_steps)
                    highest_value = total_value

        return chosen
