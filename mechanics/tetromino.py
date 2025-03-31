import numpy as np
from mechanics.colors import PIECE_COLORS

# Tetromino shapes defined as (4,4) matrices
TETROMINO_SHAPES = {
    'I': np.array([[0, 0, 0, 0],
                   [1, 1, 1, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]),
    
    'J': np.array([[1, 0, 0, 0],
                   [1, 1, 1, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]),
    
    'L': np.array([[0, 0, 1, 0],
                   [1, 1, 1, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]),
    
    'O': np.array([[1, 1, 0, 0],
                   [1, 1, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]),
    
    'S': np.array([[0, 1, 1, 0],
                   [1, 1, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]),
    
    'T': np.array([[0, 1, 0, 0],
                   [1, 1, 1, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]),
    
    'Z': np.array([[1, 1, 0, 0],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])
}

class Tetromino:
    def __init__(self, shape_name):
        self.shape_name = shape_name
        self.shape = TETROMINO_SHAPES[shape_name].copy()
        self.color = PIECE_COLORS[shape_name]
        self.row = 0
        self.col = 3  # Start from the middle of the board
        
    def rotate(self):
        """Rotate the tetromino clockwise"""
        self.shape = np.rot90(self.shape, k=-1)
        
    def move(self, delta_row, delta_col):
        """Move the tetromino by the specified amount"""
        self.row += delta_row
        self.col += delta_col
        
    def get_positions(self):
        """Get the absolute positions of the tetromino's blocks"""
        positions = []
        for i in range(4):
            for j in range(4):
                if self.shape[i][j]:
                    positions.append((self.row + i, self.col + j))
        return positions
    
    def reset_position(self):
        """Reset the tetromino to the top of the board"""
        self.row = 0
        self.col = 3 