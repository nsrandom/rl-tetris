import pygame
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechanics.tetris import TetrisGame
# from random_player import RandomPlayer
from rl_player import RLPlayer, TrainRLPlayer

from mechanics.main import setup_pygame, draw_board

def play_game_fast(player):
    game = TetrisGame()

    while not game.game_over:        
        rotations, lr_steps = player.choose_action(game)
        
        # Agent rotates and moves the piece left-right
        for _ in range(rotations):
            game.rotate_piece()
        game.move_piece(0, lr_steps)
        # And then we hard drop
        while game.move_piece(1, 0):
            pass

        # Spawn a new piece if needed
        game.spawn_piece()

    # Display final board position
    print(f"Game over! Score: {game.score}")
    screen, font = setup_pygame()
    draw_board(screen, font, game)
    while True:
        # Display the board until the user closes the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    return game.score


def play_game_slowly(player):
    screen, font = setup_pygame()

    game = TetrisGame()
    clock = pygame.time.Clock()
    fall_time = 0
    frame_count = 0
    
    print("Game initialized")  # Debug print
    
    while True:
        frame_count += 1
        if frame_count % 60 == 0:  # Print debug info every 60 frames
            print(f"Frame {frame_count}, FPS: {clock.get_fps():.1f}")
        
        # Get the time since last tick
        delta_time = clock.tick(60)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.KEYDOWN and not game.game_over:
                if event.key == pygame.K_TAB:
                    # Ask the agent for the next move
                    rotations, lr_steps = player.choose_action(game)
                    # Agent rotates and moves the piece left-right
                    for _ in range(rotations):
                        game.rotate_piece()
                    game.move_piece(0, lr_steps)
                    # Now hard drop
                    while game.move_piece(1, 0):
                        pass
                # Also allow manual control
                elif event.key == pygame.K_LEFT:
                    game.move_piece(0, -1)
                elif event.key == pygame.K_RIGHT:
                    game.move_piece(0, 1)
                elif event.key == pygame.K_DOWN:
                    game.move_piece(1, 0)
                elif event.key == pygame.K_UP:
                    game.rotate_piece()
                elif event.key == pygame.K_SPACE:
                    # Hard drop
                    while game.move_piece(1, 0):
                        pass

        # Handle automatic falling
        if not game.game_over:
            fall_time += delta_time
            if fall_time >= game.get_next_move_delay():
                game.move_piece(1, 0)
                fall_time = 0

            # Spawn new piece if needed
            game.spawn_piece()
        
        # Draw the game
        draw_board(screen, font, game)


if __name__ == "__main__":
    player = RLPlayer()

    FILE = "./agent/rlplayer.pth"
    TrainRLPlayer.load(player, FILE)
    TrainRLPlayer.train(player=player, epochs=20, batch_size=100, lr=0.1, discount=1.0)
    TrainRLPlayer.save(player, FILE)

    # play_game_slowly(player)
    play_game_fast(player)
