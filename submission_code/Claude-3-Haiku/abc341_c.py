# Read input
H, W, N = map(int, input().split())
T = input()
grid = [input() for _ in range(H)]

# Define helper functions
def is_valid(x, y):
    return 0 <= x < H and 0 <= y < W and grid[x][y] == '.'

# Simulate Takahashi's movements
x, y = 0, 0
for move in T:
    if move == 'L':
        y -= 1
    elif move == 'R':
        y += 1
    elif move == 'U':
        x -= 1
    else:
        x += 1
    if is_valid(x, y):
        continue
    else:
        break

# Find all possible current positions
possible_positions = set()
for i in range(H):
    for j in range(W):
        if is_valid(i, j):
            x, y = i, j
            for move in T:
                if move == 'L':
                    y -= 1
                elif move == 'R':
                    y += 1
                elif move == 'U':
                    x -= 1
                else:
                    x += 1
                if is_valid(x, y):
                    possible_positions.add((x, y))
                else:
                    break

# Print the answer
print(len(possible_positions))