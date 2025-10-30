H, W, N = map(int, input().split())
bars = []
for i in range(N):
    R, C, L = map(int, input().split())
    bars.append([R, C, L])

# Create a grid to track which bar occupies each cell
grid = [[0] * (W + 1) for _ in range(H + 1)]

# Initially place all bars
for i in range(N):
    R, C, L = bars[i]
    for j in range(L):
        grid[R][C + j] = i + 1

# Simulate the falling
while True:
    moved = False
    for i in range(N):
        R, C, L = bars[i]
        if R < H:  # Not on the bottom row
            can_move = True
            for j in range(L):
                if grid[R + 1][C + j] != 0:
                    can_move = False
                    break
            if can_move:
                # Move the bar down
                for j in range(L):
                    grid[R][C + j] = 0
                for j in range(L):
                    grid[R + 1][C + j] = i + 1
                bars[i][0] = R + 1
                moved = True
    if not moved:
        break

# Output the final row positions
for i in range(N):
    print(bars[i][0])