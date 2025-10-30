# YOUR CODE HERE
H, W = map(int, input().split())
r, c = map(int, input().split())
grid = [input() for _ in range(H)]
moves = input()

for move in moves:
    if move == 'U':
        if r > 1 and grid[r - 2][c - 1] == '.':
            r -= 1
    elif move == 'D':
        if r < H and grid[r][c - 1] == '.':
            r += 1
    elif move == 'L':
        if c > 1 and grid[r - 1][c - 2] == '.':
            c -= 1
    elif move == 'R':
        if c < W and grid[r - 1][c] == '.':
            c += 1

print(r, c)