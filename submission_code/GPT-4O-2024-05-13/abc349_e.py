# YOUR CODE HERE
def check_winner(grid, color_grid, color):
    # Check rows
    for i in range(3):
        if all(color_grid[i][j] == color for j in range(3)):
            return True
    # Check columns
    for j in range(3):
        if all(color_grid[i][j] == color for i in range(3)):
            return True
    # Check diagonals
    if all(color_grid[i][i] == color for i in range(3)):
        return True
    if all(color_grid[i][2-i] == color for i in range(3)):
        return True
    return False

def game_result(grid):
    # Initialize color grid
    color_grid = [['white' for _ in range(3)] for _ in range(3)]
    takahashi_score = 0
    aoki_score = 0
    moves = [(i, j) for i in range(3) for j in range(3)]
    moves.sort(key=lambda x: -grid[x[0]][x[1]])  # Sort moves by value in descending order

    for turn, (i, j) in enumerate(moves):
        if turn % 2 == 0:  # Takahashi's turn
            takahashi_score += grid[i][j]
            color_grid[i][j] = 'red'
            if check_winner(grid, color_grid, 'red'):
                return "Takahashi"
        else:  # Aoki's turn
            aoki_score += grid[i][j]
            color_grid[i][j] = 'blue'
            if check_winner(grid, color_grid, 'blue'):
                return "Aoki"
    
    # If no winner by color, check scores
    if takahashi_score > aoki_score:
        return "Takahashi"
    else:
        return "Aoki"

# Read input
grid = []
for _ in range(3):
    grid.append(list(map(int, input().split())))

# Determine the result
result = game_result(grid)
print(result)