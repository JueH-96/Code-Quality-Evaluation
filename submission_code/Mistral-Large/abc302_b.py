import sys

def find_snuke_positions(H, W, grid):
    # Directions for moving in 8 possible directions (vertical, horizontal, diagonal)
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]

    # Function to check if the sequence 's', 'n', 'u', 'k', 'e' exists in a given direction
    def check_direction(x, y, dx, dy):
        sequence = ['s', 'n', 'u', 'k', 'e']
        for i in range(5):
            if grid[x][y] != sequence[i]:
                return False
            x += dx
            y += dy
        return True

    # Iterate over each cell in the grid
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 's':
                # Check all 8 directions
                for dx, dy in directions:
                    if check_direction(i, j, dx, dy):
                        positions = [(i + k * dx, j + k * dy) for k in range(5)]
                        return positions

    return None

# Read input
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
grid = data[2:]

# Find the positions
positions = find_snuke_positions(H, W, grid)

# Output the positions
for pos in positions:
    print(pos[0] + 1, pos[1] + 1)