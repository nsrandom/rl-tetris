import numpy as np
import random

from mechanics.tetromino import Tetromino, TETROMINO_SHAPES
from mechanics.colors import BLACK, GRAY, PIECE_COLORS

class TetrisGame:
    def __init__(self, rows=20, cols=10):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.colors = np.array([[BLACK for _ in range(cols)] for _ in range(rows)])
        self.current_piece = None
        self.game_over = False
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.spawn_piece()
        
    def spawn_piece(self):
        """Create a new random tetromino piece"""
        if self.current_piece is None:
            shape_name = random.choice(list(TETROMINO_SHAPES.keys()))
            # shape_name = '.'
            self.current_piece = Tetromino(shape_name)
            if self.check_collision():
                self.game_over = True
                
    def check_collision(self):
        """Check if the current piece collides with the board or boundaries"""
        if self.current_piece is None:
            return False
            
        for row, col in self.current_piece.get_positions():
            if (row >= self.rows or col < 0 or col >= self.cols or 
                (row >= 0 and self.board[row][col])):
                return True
        return False
        
    def lock_piece(self):
        """Lock the current piece in place"""
        for row, col in self.current_piece.get_positions():
            if row >= 0:
                self.board[row][col] = 1
                self.colors[row][col] = self.current_piece.color
        self.clear_lines()
        self.current_piece = None

    def clear_lines(self):
        """Clear completed lines and update score"""
        lines_to_clear = []
        for row in range(self.rows):
            if all(self.board[row]):
                lines_to_clear.append(row)

        if lines_to_clear:
            # print(f"Yay! Cleared {lines_to_clear} lines!")
            self.lines_cleared += len(lines_to_clear)
            self.score += self.calculate_score(len(lines_to_clear))
            self.level = 1 + self.lines_cleared // 10

            # Remove the cleared lines and add new empty lines at the top
            for row in lines_to_clear:
                # For the game board
                self.board = np.vstack((np.zeros((1, self.cols)), self.board[:row], self.board[row+1:]))
                
                # For the colors - create a row of BLACK colors with proper shape
                new_color_row = np.array([[BLACK for _ in range(self.cols)]])
                self.colors = np.vstack((new_color_row, self.colors[:row], self.colors[row+1:]))
                
    def calculate_score(self, num_lines):
        """Calculate score based on number of lines cleared and current level"""
        line_scores = {1: 100, 2: 300, 3: 500, 4: 800}
        return line_scores.get(num_lines, 0) * self.level
        
    def move_piece(self, delta_row, delta_col):
        """Move the current piece if possible"""
        if self.current_piece is None or self.game_over:
            return False
            
        self.current_piece.move(delta_row, delta_col)
        if self.check_collision():
            self.current_piece.move(-delta_row, -delta_col)
            if delta_row > 0:  # If moving down, lock the piece
                self.lock_piece()
            return False
        return True

    def rotate_piece(self):
        """Rotate the current piece if possible"""
        if self.current_piece is None or self.game_over:
            return False
            
        original_shape = self.current_piece.shape.copy()
        self.current_piece.rotate()
        if self.check_collision():
            self.current_piece.shape = original_shape
            return False
        return True
        
    def get_ghost_piece_position(self):
        """Get the position where the current piece would land"""
        if self.current_piece is None:
            return []
            
        ghost_piece = Tetromino(self.current_piece.shape_name)
        ghost_piece.shape = self.current_piece.shape.copy()
        ghost_piece.row = self.current_piece.row
        ghost_piece.col = self.current_piece.col

        # Save the current piece temporarily
        original_piece = self.current_piece
        self.current_piece = ghost_piece

        # Move ghost piece down until collision
        while not self.check_collision():
            ghost_piece.move(1, 0)

        # Move back up one step since we collided
        ghost_piece.move(-1, 0)
        
        # Get the final position
        positions = ghost_piece.get_positions()
        
        # Restore the original piece
        self.current_piece = original_piece
        
        return positions
        
    def get_next_move_delay(self):
        """Get the delay between moves based on the current level"""
        return max(50, 1000 - (self.level - 1) * 50)  # Minimum 50ms delay


@staticmethod
def apply_moves(game: TetrisGame, rotations: int, lr_steps: int):
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
