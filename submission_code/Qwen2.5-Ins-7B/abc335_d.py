# N = int(input())
N = 5

# Initialize the grid with zeros
grid = [[0 for _ in range(N)] for _ in range(N)]

# Place Takahashi at the center
center = (N // 2, N // 2)
grid[center[0]][center[1]] = 'T'

# Function to get the adjacent cells
def get_adjacent_cells(x, y):
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

# Function to place the dragon parts
def place_parts(x, y, direction, parts):
    if x < 0 or x >= N or y < 0 or y >= N or grid[x][y] != 0:
        return
    grid[x][y] = parts.pop(0)
    if parts:
        if direction == 'right':
            place_parts(x, y+1, 'right', parts)
        elif direction == 'left':
            place_parts(x, y-1, 'left', parts)
        elif direction == 'up':
            place_parts(x-1, y, 'up', parts)
        elif direction == 'down':
            place_parts(x+1, y, 'down', parts)

# Place the dragon parts
parts = list(range(1, N*N))
directions = ['right', 'down', 'left', 'up']
current_direction = 0
x, y = center
for part in parts:
    if part == 1:
        grid[x][y] = part
    else:
        if current_direction == 0:
            y += 1
        elif current_direction == 1:
            x += 1
        elif current_direction == 2:
            y -= 1
        elif current_direction == 3:
            x -= 1
        grid[x][y] = part
        if part % 2 == 0:
            current_direction = (current_direction + 1) % 4

# Print the grid
for row in grid:
    print(' '.join(map(str, row)))