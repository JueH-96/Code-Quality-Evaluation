import sys

def check_winner(grid, player):
    # Check rows, columns, and diagonals for three consecutive cells of the same color
    for i in range(3):
        if all(grid[i][j] == player for j in range(3)) or all(grid[j][i] == player for j in range(3)):
            return True
    if all(grid[i][i] == player for i in range(3)) or all(grid[i][2-i] == player for i in range(3)):
        return True
    return False

def play_game(grid):
    # Initialize the game state
    scores = [0, 0]  # Scores for Takahashi (0) and Aoki (1)
    colors = [[-1] * 3 for _ in range(3)]  # -1: white, 0: red, 1: blue
    turn = 0  # 0: Takahashi, 1: Aoki

    while True:
        # Find the best move for the current player
        best_score = -float('inf')
        best_move = None
        for i in range(3):
            for j in range(3):
                if colors[i][j] == -1:
                    new_score = scores[turn] + grid[i][j]
                    if new_score > best_score:
                        best_score = new_score
                        best_move = (i, j)

        # Apply the best move
        i, j = best_move
        scores[turn] = best_score
        colors[i][j] = turn

        # Check if the current player wins
        if check_winner(colors, turn):
            return "Takahashi" if turn == 0 else "Aoki"

        # Check if the game is a draw
        if all(colors[i][j] != -1 for i in range(3) for j in range(3)):
            return "Takahashi" if scores[0] > scores[1] else "Aoki"

        # Switch turns
        turn = 1 - turn

# Read input
grid = [list(map(int, input().split())) for _ in range(3)]

# Determine the winner
winner = play_game(grid)
print(winner)