import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechanics.tetris import TetrisGame
from random_player import RandomPlayer

class Arena:
    def __init__(self):
        self.game = TetrisGame()
        self.player = RandomPlayer()
    
    def play_game(self):
        game = self.game

        while not game.game_over:        
            rotations, lr_steps = self.player.choose_action(game)
            
            # Agent rotates and moves the piece left-right
            for _ in range(rotations):
                game.rotate_piece()
            game.move_piece(0, lr_steps)
            # And then we hard drop
            while game.move_piece(1, 0):
                pass
            
            # Spawn a new piece if needed
            game.spawn_piece()

        return game.score


if __name__ == "__main__":
    arena = Arena()
    score = arena.play_game()
    print(f"Game over! Score: {score}")

