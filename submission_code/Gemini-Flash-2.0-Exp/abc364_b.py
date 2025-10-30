H, W = map(int, input().split())
S_i, S_j = map(int, input().split())
S_i -= 1
S_j -= 1
grid = []
for _ in range(H):
    grid.append(input())
X = input()

for move in X:
    if move == 'L':
        if S_j > 0 and grid[S_i][S_j - 1] == '.':
            S_j -= 1
    elif move == 'R':
        if S_j < W - 1 and grid[S_i][S_j + 1] == '.':
            S_j += 1
    elif move == 'U':
        if S_i > 0 and grid[S_i - 1][S_j] == '.':
            S_i -= 1
    elif move == 'D':
        if S_i < H - 1 and grid[S_i + 1][S_j] == '.':
            S_i += 1

print(S_i + 1, S_j + 1)