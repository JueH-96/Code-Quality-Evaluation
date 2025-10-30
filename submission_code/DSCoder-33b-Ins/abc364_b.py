# YOUR CODE HERE
H, W = map(int, input().split())
S_i, S_j = map(int, input().split())
grid = [list(input()) for _ in range(H)]
X = input()

for move in X:
    if move == 'L':
        if S_j > 1 and grid[S_i - 1][S_j - 2] == '.':
            S_j -= 1
    elif move == 'R':
        if S_j < W and grid[S_i - 1][S_j] == '.':
            S_j += 1
    elif move == 'U':
        if S_i > 1 and grid[S_i - 2][S_j - 1] == '.':
            S_i -= 1
    elif move == 'D':
        if S_i < H and grid[S_i][S_j - 1] == '.':
            S_i += 1

print(S_i, S_j)