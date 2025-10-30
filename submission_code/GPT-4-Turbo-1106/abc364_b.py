def move_takahashi(H, W, Si, Sj, grid, X):
    # Takahashi's initial position
    current_i, current_j = Si - 1, Sj - 1

    # Process each character in X
    for move in X:
        if move == 'L' and current_j > 0 and grid[current_i][current_j - 1] == '.':
            current_j -= 1
        elif move == 'R' and current_j < W - 1 and grid[current_i][current_j + 1] == '.':
            current_j += 1
        elif move == 'U' and current_i > 0 and grid[current_i - 1][current_j] == '.':
            current_i -= 1
        elif move == 'D' and current_i < H - 1 and grid[current_i + 1][current_j] == '.':
            current_i += 1

    # Return the final position (1-indexed)
    return current_i + 1, current_j + 1

# Read input
H, W = map(int, input().split())
Si, Sj = map(int, input().split())
grid = [input() for _ in range(H)]
X = input()

# Solve the problem
final_i, final_j = move_takahashi(H, W, Si, Sj, grid, X)

# Print the answer
print(final_i, final_j)