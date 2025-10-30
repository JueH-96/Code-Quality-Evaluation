# Read the input
N, M = map(int, input().split())
grid = [input() for _ in range(N)]

# Define the directions
directions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# Function to check if a square is ice
def is_ice(i, j):
    return grid[i][j] == '.'

# Function to move in a direction
def move(i, j, di, dj):
    while 0 <= i < N and 0 <= j < M and is_ice(i, j):
        i += di
        j += dj
    return i, j

# Count the number of ice squares the player can touch
count = 0
i, j = 2, 2
for di, dj in directions.values():
    ni, nj = move(i, j, di, dj)
    if (i, j) != (ni, nj):
        count += 1
        i, j = ni, nj
print(count)