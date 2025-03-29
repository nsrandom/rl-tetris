import pygame
import sys
from tetris import TetrisGame
from colors import BLACK, GRAY, WHITE

# Initialize Pygame
pygame.init()

# Constants
BLOCK_SIZE = 30
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
# Note: 10 rows and 20 columns is the standard Tetris board size
BOARD_OFFSET_X = (SCREEN_WIDTH - 10 * BLOCK_SIZE) // 2
BOARD_OFFSET_Y = (SCREEN_HEIGHT - 20 * BLOCK_SIZE) // 2

# Create the game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tetris")

# Initialize fonts
font = pygame.font.Font(None, 36)

def draw_block(x, y, color):
    """Draw a single block with a 3D effect"""
    pygame.draw.rect(screen, color, (x, y, BLOCK_SIZE, BLOCK_SIZE))
    pygame.draw.rect(screen, WHITE, (x, y, BLOCK_SIZE, BLOCK_SIZE), 1)
    
def draw_board(game):
    """Draw the game board"""
    # Draw background
    screen.fill(BLACK)
    
    # Draw board outline
    pygame.draw.rect(screen, WHITE, 
                    (BOARD_OFFSET_X - 2, BOARD_OFFSET_Y - 2,
                     game.cols * BLOCK_SIZE + 4, game.rows * BLOCK_SIZE + 4), 2)
    
    # Draw fixed blocks
    for row in range(game.rows):
        for col in range(game.cols):
            if game.board[row][col]:
                x = BOARD_OFFSET_X + col * BLOCK_SIZE
                y = BOARD_OFFSET_Y + row * BLOCK_SIZE
                draw_block(x, y, game.colors[row][col])
                
    # Draw ghost piece
    if not game.game_over and game.current_piece:
        ghost_positions = game.get_ghost_piece_position()
        for row, col in ghost_positions:
            if row >= 0:
                x = BOARD_OFFSET_X + col * BLOCK_SIZE
                y = BOARD_OFFSET_Y + row * BLOCK_SIZE
                pygame.draw.rect(screen, GRAY, (x, y, BLOCK_SIZE, BLOCK_SIZE), 1)
    
    # Draw current piece
    if game.current_piece:
        for row, col in game.current_piece.get_positions():
            if row >= 0:
                x = BOARD_OFFSET_X + col * BLOCK_SIZE
                y = BOARD_OFFSET_Y + row * BLOCK_SIZE
                draw_block(x, y, game.current_piece.color)
                
    # Draw score and level
    score_text = font.render(f"Score: {game.score}", True, WHITE)
    level_text = font.render(f"Level: {game.level}", True, WHITE)
    lines_text = font.render(f"Lines: {game.lines_cleared}", True, WHITE)
    
    screen.blit(score_text, (20, 20))
    screen.blit(level_text, (20, 60))
    screen.blit(lines_text, (20, 100))
    
    if game.game_over:
        game_over_text = font.render("GAME OVER", True, WHITE)
        text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(game_over_text, text_rect)
        
    pygame.display.flip()

def main():
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
                if event.key == pygame.K_LEFT:
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
        draw_board(game)

if __name__ == "__main__":
    main() 