# YOUR CODE HERE
import sys
from itertools import permutations

def check_winner(grid):
    # Check rows and columns
    for i in range(3):
        if grid[i][0] == grid[i][1] == grid[i][2] and grid[i][0] != 0:
            return grid[i][0]
        if grid[0][i] == grid[1][i] == grid[2][i] and grid[0][i] != 0:
            return grid[0][i]
    
    # Check diagonals
    if grid[0][0] == grid[1][1] == grid[2][2] and grid[0][0] != 0:
        return grid[0][0]
    if grid[0][2] == grid[1][1] == grid[2][0] and grid[0][2] != 0:
        return grid[0][2]
    
    return 0

def minimax(grid, depth, is_maximizing, alpha, beta):
    winner = check_winner(grid)
    if winner == 1:
        return 1
    if winner == -1:
        return -1
    if all(grid[i][j] != 0 for i in range(3) for j in range(3)):
        return 0
    
    if is_maximizing:
        max_eval = -float('inf')
        for i in range(3):
            for j in range(3):
                if grid[i][j] == 0:
                    grid[i][j] = 1
                    eval = minimax(grid, depth + 1, False, alpha, beta)
                    grid[i][j] = 0
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if grid[i][j] == 0:
                    grid[i][j] = -1
                    eval = minimax(grid, depth + 1, True, alpha, beta)
                    grid[i][j] = 0
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
        return min_eval

def main():
    grid = []
    for _ in range(3):
        row = list(map(int, input().split()))
        grid.append(row)
    
    if minimax(grid, 0, True, -float('inf'), float('inf')) >= 0:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()