# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
grid = data[2:]

# Directions for vertical, horizontal, and diagonal movements
directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

def find_snukes():
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 's':
                for direction in directions:
                    if check_direction(i, j, direction):
                        return

def check_direction(start_row, start_col, direction):
    row, col = start_row, start_col
    sequence = 'snukes'
    positions = []

    for char in sequence:
        if 0 <= row < H and 0 <= col < W and grid[row][col] == char:
            positions.append((row + 1, col + 1))
            row += direction[0]
            col += direction[1]
        else:
            return False

    for pos in positions:
        print(pos[0], pos[1])
    return True

find_snukes()