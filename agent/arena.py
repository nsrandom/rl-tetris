import pygame
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechanics.tetris import TetrisGame, apply_moves
# from policy_gradient_player import PolicyGradient_RLPlayer
# from train_rl_player import TrainRLPlayer
from dqn_rl_player import DQN_RLPlayer, TrainRLPlayer

from mechanics.main import setup_pygame, draw_board


def benchmark_player(player, num_times=100):
    scores = []
    for i in range(num_times):
        game = TetrisGame()
        while not game.game_over:
            rotations, lr_steps = player.choose_action(game)
            apply_moves(game, rotations, lr_steps)
            game.spawn_piece()
        scores.append(game.score)
        print(f"Game {i+1}: Score {game.score}")

    scores = np.array([scores])

    print(f"After {num_times} runs:")
    print(f"\tMean score: {np.mean(scores)}")
    print(f"\tMedian score: {np.median(scores)}")
    print(f"\tBest score: {np.max(scores)}")
    print(f"\tStandard deviation: {np.std(scores)}")
    print(f"\t25th percentile score: {np.percentile(scores, 25)}")
    print(f"\t75th percentile score: {np.percentile(scores, 75)}")

def play_game_fast(player):
    game = TetrisGame()

    while not game.game_over:
        # Ask the agent for the next move, and apply it
        rotations, lr_steps = player.choose_action(game)
        apply_moves(game, rotations, lr_steps)

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
    
    print("Game initialized")  # Debug print
    
    while True:
        # Get the time since last tick
        delta_time = clock.tick(60)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN and not game.game_over:
                if event.key == pygame.K_TAB:
                    # Ask the agent for the next move, and apply it
                    rotations, lr_steps = player.choose_action(game)
                    apply_moves(game, rotations, lr_steps)

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

        # Spawn new piece if needed
        game.spawn_piece()

        # Handle automatic falling
        if not game.game_over:
            fall_time += delta_time
            if fall_time >= game.get_next_move_delay():
                # Ask the agent for the next move, and apply it
                rotations, lr_steps = player.choose_action(game)
                apply_moves(game, rotations, lr_steps)
                fall_time = 0

            # Spawn new piece if needed
            game.spawn_piece()
        
        # Draw the game
        draw_board(screen, font, game)


if __name__ == "__main__":
    # player = PolicyGradient_RLPlayer(explore_prob=0.2)
    player = DQN_RLPlayer(explore_prob=0.1)

    # Load the player model from disk
    FILE = "./agent/dqn_player.pth"
    TrainRLPlayer.load(player, FILE)

    # Train the player, and save the results
    # TrainRLPlayer.train(player=player, epochs=1000, batch_size=50, lr=0.001, discount=0.9)
    # TrainRLPlayer.save(player, FILE)

    # Don't explore any more
    player.explore_prob = 0.0

    benchmark_player(player)
    # play_game_slowly(player)
    # play_game_fast(player)

    
