# YOUR CODE HERE
H, W = map(int, input().split())
Si, Sj = map(int, input().split())
grid = [input() for _ in range(H)]
X = input()

i, j = Si - 1, Sj - 1

for move in X:
    if move == 'L' and j > 0 and grid[i][j - 1] == '.':
        j -= 1
    elif move == 'R' and j < W - 1 and grid[i][j + 1] == '.':
        j += 1
    elif move == 'U' and i > 0 and grid[i - 1][j] == '.':
        i -= 1
    elif move == 'D' and i < H - 1 and grid[i + 1][j] == '.':
        i += 1

print(i + 1, j + 1)