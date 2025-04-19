import pygame
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechanics.tetris import TetrisGame
# from random_player import RandomPlayer
from rl_player import RLPlayer

from mechanics.main import setup_pygame, draw_board

class Arena:
    def __init__(self):
        self.game = TetrisGame()
        self.player = RLPlayer()
        # self.player = RandomPlayer()
    
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

    # Display ending board state
    display = True
    if display:
        screen, font = setup_pygame()
        draw_board(screen, font, arena.game)
        while True:
            # Display the board until the user closes the window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
