H, W, N = map(int, input().split())
T = input().strip()
grid = []
for _ in range(H):
    s = input().strip()
    grid.append(s)

# Precompute land cells
land = set()
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            land.add((i, j))

current_positions = land.copy()

for step in T:
    next_positions = set()
    for (i, j) in current_positions:
        if step == 'L':
            ni, nj = i, j - 1
        elif step == 'R':
            ni, nj = i, j + 1
        elif step == 'U':
            ni, nj = i - 1, j
        elif step == 'D':
            ni, nj = i + 1, j
        else:
            ni, nj = i, j  # This case shouldn't happen as per problem constraints
        if (ni, nj) in land:
            next_positions.add((ni, nj))
    current_positions = next_positions
    if not current_positions:
        break  # Though the problem states there's at least one, so this is just a safeguard

print(len(current_positions))