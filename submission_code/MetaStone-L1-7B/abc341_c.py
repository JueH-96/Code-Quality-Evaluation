H, W, N = map(int, input().split())
T = input().strip()
grid = []
for _ in range(H):
    s = input().strip()
    grid.append([c == '.' for c in s])

# Precompute the move deltas
moves = []
for c in T:
    if c == 'L':
        di, dj = 0, -1
    elif c == 'R':
        di, dj = 0, 1
    elif c == 'U':
        di, dj = -1, 0
    elif c == 'D':
        di, dj = 1, 0
    moves.append((di, dj))

count = 0

for i in range(H):
    for j in range(W):
        if not grid[i][j]:
            continue  # Starting cell is not land, skip
        current_i, current_j = i, j
        valid = True
        for (di, dj) in moves:
            next_i = current_i + di
            next_j = current_j + dj
            # Check if next is within grid
            if not (0 <= next_i < H and 0 <= next_j < W):
                valid = False
                break
            if not grid[next_i][next_j]:
                valid = False
                break
            current_i, current_j = next_i, next_j
        if valid:
            count += 1

print(count)