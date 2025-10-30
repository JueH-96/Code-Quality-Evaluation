from itertools import permutations

# Function to rotate a polyomino 90 degrees clockwise
def rotate(polyomino):
    return list(zip(*polyomino[::-1]))

# Function to check if a polyomino fits in the grid at a given position
def fits_in_grid(polyomino, grid, row, col):
    for i in range(len(polyomino)):
        for j in range(len(polyomino[0])):
            if polyomino[i][j] == '#' and (row + i >= 4 or col + j >= 4 or grid[row + i][col + j] != '.'):
                return False
    return True

# Function to place a polyomino in the grid at a given position
def place_in_grid(polyomino, grid, row, col):
    for i in range(len(polyomino)):
        for j in range(len(polyomino[0])):
            if polyomino[i][j] == '#':
                grid[row + i][col + j] = '#'

# Function to remove a polyomino from the grid at a given position
def remove_from_grid(polyomino, grid, row, col):
    for i in range(len(polyomino)):
        for j in range(len(polyomino[0])):
            if polyomino[i][j] == '#':
                grid[row + i][col + j] = '.'

# Function to check if the grid is completely filled with polyominoes
def is_grid_filled(grid):
    return all(all(cell == '#' for cell in row) for row in grid)

# Recursive function to try to fit polyominoes in the grid
def solve(polyominoes, grid, index):
    if index == len(polyominoes):
        return is_grid_filled(grid)
    
    polyomino = polyominoes[index]
    for _ in range(4):  # Try all four rotations
        for row in range(4):
            for col in range(4):
                if fits_in_grid(polyomino, grid, row, col):
                    place_in_grid(polyomino, grid, row, col)
                    if solve(polyominoes, grid, index + 1):
                        return True
                    remove_from_grid(polyomino, grid, row, col)
        polyomino = rotate(polyomino)  # Rotate the polyomino for the next iteration
    
    return False

# Read input
polyominoes = []
for _ in range(3):
    polyomino = [list(input()) for _ in range(4)]
    polyominoes.append(polyomino)

# Try all permutations of the polyominoes
for perm in permutations(polyominoes):
    grid = [['.' for _ in range(4)] for _ in range(4)]
    if solve(list(perm), grid, 0):
        print("Yes")
        break
else:
    print("No")