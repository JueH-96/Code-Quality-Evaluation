H, W, N = map(int, input().split())
T = input().strip()
grid = [input().strip() for _ in range(H)]

# Precompute directions as a list of tuples (dx, dy)
directions = []
for c in T:
    if c == 'L':
        directions.append((0, -1))
    elif c == 'R':
        directions.append((0, 1))
    elif c == 'U':
        directions.append((-1, 0))
    elif c == 'D':
        directions.append((1, 0))

# Precompute grid as a boolean grid (True for land)
grid_bool = [[cell == '.' for cell in row] for row in grid]

# Collect starting positions (land cells not on perimeter)
start_positions = []
for i in range(1, H + 1):
    for j in range(1, W + 1):
        if i == 1 or i == H or j == 1 or j == W:
            continue
        if grid_bool[i - 1][j - 1]:
            start_positions.append((i, j))

result = set()

for x, y in start_positions:
    current_x, current_y = x, y
    valid = True
    for dx, dy in directions:
        current_x += dx
        current_y += dy
        # Check if current position is within bounds and is land
        if not (1 <= current_x <= H and 1 <= current_y <= W):
            valid = False
            break
        if not grid_bool[current_x - 1][current_y - 1]:
            valid = False
            break
    if valid:
        result.add((current_x, current_y))

print(len(result))