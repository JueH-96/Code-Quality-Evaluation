import sys
from functools import lru_cache

def main():
    A = [list(map(int, sys.stdin.readline().split())) for _ in range(3)]
    
    def has_line(grid, player):
        # Check rows
        for i in range(3):
            if all(grid[i][j] == player for j in range(3)):
                return True
        # Check columns
        for j in range(3):
            if all(grid[i][j] == player for i in range(3)):
                return True
        # Check diagonals
        if all(grid[i][i] == player for i in range(3)):
            return True
        if all(grid[i][2 - i] == player for i in range(3)):
            return True
        return False
    
    def calculate_scores(grid_tuple):
        t_score = 0
        a_score = 0
        for i in range(3):
            for j in range(3):
                cell = grid_tuple[i][j]
                if cell == 1:
                    t_score += A[i][j]
                elif cell == 2:
                    a_score += A[i][j]
        return t_score, a_score
    
    @lru_cache(maxsize=None)
    def game_state_result(grid_tuple):
        # Count filled cells
        filled = 0
        for i in range(3):
            for j in range(3):
                if grid_tuple[i][j] != 0:
                    filled += 1
        current_player = 1 if (filled % 2 == 0) else 2
        
        # Generate empty cells
        empty_cells = []
        for i in range(3):
            for j in range(3):
                if grid_tuple[i][j] == 0:
                    empty_cells.append( (i, j) )
        
        # For each possible move
        for i, j in empty_cells:
            # Create new grid
            new_grid = [ list(row) for row in grid_tuple ]
            new_grid[i][j] = current_player
            new_grid_tuple = ( tuple(new_grid[0]), tuple(new_grid[1]), tuple(new_grid[2]) )
            
            # Check if this move causes a line
            if has_line(new_grid_tuple, current_player):
                # current player wins
                return 'T' if current_player == 1 else 'A'
            
            # Check if grid is full
            new_filled = filled + 1
            if new_filled == 9:
                t_score, a_score = calculate_scores(new_grid_tuple)
                if t_score > a_score:
                    return 'T'
                else:
                    return 'A'
            
            # Recurse
            next_result = game_state_result(new_grid_tuple)
            
            # Check if current player can win with this move
            if current_player == 1:
                if next_result == 'T':
                    return 'T'
            else:
                if next_result == 'A':
                    return 'A'
        
        # If no moves lead to a win, return the opposite
        return 'T' if current_player == 2 else 'A'
    
    initial_grid = ( (0, 0, 0), (0, 0, 0), (0, 0, 0) )
    result = game_state_result(initial_grid)
    print("Takahashi" if result == 'T' else "Aoki")

if __name__ == "__main__":
    main()