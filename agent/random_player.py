import random

class RandomPlayer:
    def choose_action(self, game):
        rotations = random.randint(0, 3)
        lr_steps = random.randint(-1, 1)
        return rotations, lr_steps

