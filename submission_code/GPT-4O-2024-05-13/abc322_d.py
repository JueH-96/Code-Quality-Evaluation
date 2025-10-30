# YOUR CODE HERE
import sys
from itertools import permutations

input = sys.stdin.read
data = input().strip().split()

# Read the polyominoes
polyominoes = []
for i in range(3):
    polyomino = [data[i*4 + j] for j in range(4)]
    polyominoes.append(polyomino)

# Function to rotate a polyomino 90 degrees clockwise
def rotate(polyomino):
    return [''.join(row[i] for row in polyomino[::-1]) for i in range(4)]

# Function to get all rotations of a polyomino
def get_rotations(polyomino):
    rotations = [polyomino]
    for _ in range(3):
        polyomino = rotate(polyomino)
        rotations.append(polyomino)
    return rotations

# Function to get the bounding box of a polyomino
def get_bounding_box(polyomino):
    min_row, max_row = 4, -1
    min_col, max_col = 4, -1
    for r in range(4):
        for c in range(4):
            if polyomino[r][c] == '#':
                min_row = min(min_row, r)
                max_row = max(max_row, r)
                min_col = min(min_col, c)
                max_col = max(max_col, c)
    return min_row, max_row, min_col, max_col

# Function to get all valid placements of a polyomino on the grid
def get_placements(polyomino):
    placements = []
    min_row, max_row, min_col, max_col = get_bounding_box(polyomino)
    for r in range(4 - (max_row - min_row)):
        for c in range(4 - (max_col - min_col)):
            placement = [['.' for _ in range(4)] for _ in range(4)]
            for i in range(min_row, max_row + 1):
                for j in range(min_col, max_col + 1):
                    if polyomino[i][j] == '#':
                        placement[r + i - min_row][c + j - min_col] = '#'
            placements.append(placement)
    return placements

# Function to check if two placements overlap
def overlap(p1, p2):
    for r in range(4):
        for c in range(4):
            if p1[r][c] == '#' and p2[r][c] == '#':
                return True
    return False

# Function to check if three placements cover the entire grid
def covers_grid(p1, p2, p3):
    grid = [['.' for _ in range(4)] for _ in range(4)]
    for r in range(4):
        for c in range(4):
            if p1[r][c] == '#':
                grid[r][c] = '#'
            if p2[r][c] == '#':
                grid[r][c] = '#'
            if p3[r][c] == '#':
                grid[r][c] = '#'
    for r in range(4):
        for c in range(4):
            if grid[r][c] == '.':
                return False
    return True

# Get all rotations and placements for each polyomino
all_placements = []
for polyomino in polyominoes:
    rotations = get_rotations(polyomino)
    placements = []
    for rotation in rotations:
        placements.extend(get_placements(rotation))
    all_placements.append(placements)

# Try all combinations of placements
for p1 in all_placements[0]:
    for p2 in all_placements[1]:
        if overlap(p1, p2):
            continue
        for p3 in all_placements[2]:
            if overlap(p1, p3) or overlap(p2, p3):
                continue
            if covers_grid(p1, p2, p3):
                print("Yes")
                sys.exit(0)

print("No")