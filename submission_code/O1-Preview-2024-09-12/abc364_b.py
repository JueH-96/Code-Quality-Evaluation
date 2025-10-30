# YOUR CODE HERE
H, W = map(int, input().split())
S_i, S_j = map(int, input().split())
grid = [input() for _ in range(H)]
X = input()

# Convert S_i and S_j to 0-based indices
i, j = S_i - 1, S_j - 1

for c in X:
    if c == 'L':
        if j > 0 and grid[i][j - 1] == '.':
            j -= 1
    elif c == 'R':
        if j + 1 < W and grid[i][j + 1] == '.':
            j += 1
    elif c == 'U':
        if i > 0 and grid[i - 1][j] == '.':
            i -= 1
    elif c == 'D':
        if i + 1 < H and grid[i + 1][j] == '.':
            i += 1

# Convert back to 1-based indices for output
print(i + 1, j + 1)