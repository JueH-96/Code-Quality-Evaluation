# Read the grid values
grid_values = []
for _ in range(3):
    row = list(map(int, input().split()))
    grid_values.extend(row)

def check_win(grid, player):
    cell = 1 if player == 'T' else 2
    # Check rows
    for i in range(0, 9, 3):
        if grid[i] == cell and grid[i+1] == cell and grid[i+2] == cell:
            return True
    # Check columns
    for i in range(3):
        if grid[i] == cell and grid[i+3] == cell and grid[i+6] == cell:
            return True
    # Check diagonals
    if grid[0] == cell and grid[4] == cell and grid[8] == cell:
        return True
    if grid[2] == cell and grid[4] == cell and grid[6] == cell:
        return True
    return False

from functools import lru_cache

@lru_cache(maxsize=None)
def evaluate(grid):
    # Check if all cells are filled
    if sum(1 for cell in grid if cell != 0) == 9:
        t_score = sum(grid_values[i] for i in range(9) if grid[i] == 1)
        a_score = sum(grid_values[i] for i in range(9) if grid[i] == 2)
        return t_score - a_score

    taken = sum(1 for cell in grid if cell != 0)
    current_player = 'T' if (taken % 2 == 0) else 'A'

    possible_moves = [i for i in range(9) if grid[i] == 0]

    if not possible_moves:
        return 0  # This case shouldn't occur as per problem statement

    if current_player == 'T':
        max_val = -float('inf')
        for i in possible_moves:
            new_grid = list(grid)
            new_grid[i] = 1
            new_grid = tuple(new_grid)
            if check_win(new_grid, 'T'):
                return float('inf')
            current = grid_values[i] + evaluate(new_grid)
            if current > max_val:
                max_val = current
        return max_val if max_val != -float('inf') else 0
    else:
        min_val = float('inf')
        for i in possible_moves:
            new_grid = list(grid)
            new_grid[i] = 2
            new_grid = tuple(new_grid)
            if check_win(new_grid, 'A'):
                return -float('inf')
            current = -grid_values[i] + evaluate(new_grid)
            if current < min_val:
                min_val = current
        return min_val if min_val != float('inf') else 0

initial_grid = tuple([0] * 9)
result = evaluate(initial_grid)

if result > 0:
    print("Takahashi")
else:
    print("Aoki")