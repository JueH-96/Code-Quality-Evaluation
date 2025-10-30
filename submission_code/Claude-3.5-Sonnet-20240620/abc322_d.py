# YOUR CODE HERE
import sys
from itertools import product

def rotate_90(piece):
    return [''.join(row) for row in zip(*piece[::-1])]

def all_rotations(piece):
    rotations = [piece]
    for _ in range(3):
        rotations.append(rotate_90(rotations[-1]))
    return rotations

def can_place(grid, piece, x, y):
    for i, row in enumerate(piece):
        for j, cell in enumerate(row):
            if cell == '#':
                if x+i >= 4 or y+j >= 4 or grid[x+i][y+j] != '.':
                    return False
    return True

def place(grid, piece, x, y):
    for i, row in enumerate(piece):
        for j, cell in enumerate(row):
            if cell == '#':
                grid[x+i][y+j] = '#'

def remove(grid, piece, x, y):
    for i, row in enumerate(piece):
        for j, cell in enumerate(row):
            if cell == '#':
                grid[x+i][y+j] = '.'

def solve(grid, pieces):
    if not pieces:
        return all(cell == '#' for row in grid for cell in row)
    
    piece = pieces[0]
    for rotation in all_rotations(piece):
        for x, y in product(range(4), repeat=2):
            if can_place(grid, rotation, x, y):
                place(grid, rotation, x, y)
                if solve(grid, pieces[1:]):
                    return True
                remove(grid, rotation, x, y)
    return False

pieces = [[] for _ in range(3)]
for i in range(3):
    for _ in range(4):
        pieces[i].append(sys.stdin.readline().strip())

grid = [['.' for _ in range(4)] for _ in range(4)]

print("Yes" if solve(grid, pieces) else "No")