H, W, N = map(int, input().split())
T = input().strip()

# Generate reversed and inverted steps
reversed_T = T[::-1]
reversed_steps = []
for c in reversed_T:
    if c == 'L':
        reversed_steps.append('R')
    elif c == 'R':
        reversed_steps.append('L')
    elif c == 'U':
        reversed_steps.append('D')
    elif c == 'D':
        reversed_steps.append('U')
reversed_steps = ''.join(reversed_steps)

# Read the grid
grid = [input().strip() for _ in range(H)]

count = 0

for i in range(H):
    for j in range(W):
        if grid[i][j] != '.':
            continue
        current_i, current_j = i, j
        valid = True
        for c in reversed_steps:
            if c == 'L':
                new_i, new_j = current_i, current_j - 1
            elif c == 'R':
                new_i, new_j = current_i, current_j + 1
            elif c == 'U':
                new_i, new_j = current_i - 1, current_j
            elif c == 'D':
                new_i, new_j = current_i + 1, current_j
            # Check boundaries
            if not (0 <= new_i < H and 0 <= new_j < W):
                valid = False
                break
            # Check if cell is land
            if grid[new_i][new_j] != '.':
                valid = False
                break
            current_i, current_j = new_i, new_j
        if valid:
            count += 1

print(count)