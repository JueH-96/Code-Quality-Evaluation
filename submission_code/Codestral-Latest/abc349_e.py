# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

# Parse the input grid
grid = [[int(data[i*3 + j]) for j in range(3)] for i in range(3)]

# Function to check if a player can win by forming a line of three consecutive cells of the same color
def can_win(grid, color):
    # Check rows and columns
    for i in range(3):
        if all(grid[i][j] == color for j in range(3)) or all(grid[j][i] == color for j in range(3)):
            return True
    # Check diagonals
    if grid[0][0] == grid[1][1] == grid[2][2] == color or grid[0][2] == grid[1][1] == grid[2][0] == color:
        return True
    return False

# Function to simulate the game and determine the winner
def determine_winner(grid):
    # Total sum of the grid
    total_sum = sum(sum(row) for row in grid)

    # If the total sum is odd, Takahashi goes first
    if total_sum % 2 != 0:
        # Takahashi's first move
        for i in range(3):
            for j in range(3):
                if grid[i][j] != 0:
                    grid[i][j] = 'T'
                    if can_win(grid, 'T'):
                        return "Takahashi"
                    grid[i][j] = 0
        return "Aoki"
    else:
        # Aoki's first move
        for i in range(3):
            for j in range(3):
                if grid[i][j] != 0:
                    grid[i][j] = 'A'
                    if can_win(grid, 'A'):
                        return "Aoki"
                    grid[i][j] = 0
        return "Takahashi"

# Determine the winner and print the result
winner = determine_winner(grid)
print(winner)