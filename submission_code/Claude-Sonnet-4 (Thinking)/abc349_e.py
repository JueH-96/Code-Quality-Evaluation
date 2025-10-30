def check_winner(grid):
    # Check rows
    for i in range(3):
        if grid[i][0] == grid[i][1] == grid[i][2] and grid[i][0] != 0:
            return grid[i][0]
    
    # Check columns
    for j in range(3):
        if grid[0][j] == grid[1][j] == grid[2][j] and grid[0][j] != 0:
            return grid[0][j]
    
    # Check diagonals
    if grid[0][0] == grid[1][1] == grid[2][2] and grid[0][0] != 0:
        return grid[0][0]
    if grid[0][2] == grid[1][1] == grid[2][0] and grid[0][2] != 0:
        return grid[0][2]
    
    return 0

def minimax(grid, values, takahashi_score, aoki_score, is_takahashi_turn):
    # Check if someone already won
    winner = check_winner(grid)
    if winner == 1:  # Takahashi won
        return 1
    elif winner == 2:  # Aoki won
        return -1
    
    # Find white cells
    white_cells = []
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 0:
                white_cells.append((i, j))
    
    if not white_cells:
        # Game over, compare scores
        if takahashi_score > aoki_score:
            return 1
        else:
            return -1  # aoki_score > takahashi_score (sum is odd, so no tie)
    
    if is_takahashi_turn:
        best_value = -2
        for i, j in white_cells:
            new_grid = [row[:] for row in grid]
            new_grid[i][j] = 1
            value = minimax(new_grid, values, takahashi_score + values[i][j], aoki_score, False)
            best_value = max(best_value, value)
        return best_value
    else:
        best_value = 2
        for i, j in white_cells:
            new_grid = [row[:] for row in grid]
            new_grid[i][j] = 2
            value = minimax(new_grid, values, takahashi_score, aoki_score + values[i][j], True)
            best_value = min(best_value, value)
        return best_value

# Read input
values = []
for _ in range(3):
    row = list(map(int, input().split()))
    values.append(row)

# Initialize game state
grid = [[0 for _ in range(3)] for _ in range(3)]

# Determine winner
result = minimax(grid, values, 0, 0, True)

if result > 0:
    print("Takahashi")
else:
    print("Aoki")