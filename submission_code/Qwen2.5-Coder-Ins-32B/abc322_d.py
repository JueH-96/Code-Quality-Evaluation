# YOUR CODE HERE
from itertools import permutations

def rotate_polyomino(polyomino):
    return [''.join(row) for row in zip(*polyomino[::-1])]

def get_all_rotations(polyomino):
    rotations = []
    for _ in range(4):
        polyomino = rotate_polyomino(polyomino)
        rotations.append(polyomino)
    return rotations

def can_place(grid, polyomino, x, y):
    for i in range(4):
        for j in range(4):
            if polyomino[i][j] == '#' and (x + i >= 4 or y + j >= 4 or grid[x + i][y + j] == '#'):
                return False
    return True

def place_polyomino(grid, polyomino, x, y):
    for i in range(4):
        for j in range(4):
            if polyomino[i][j] == '#':
                grid[x + i][y + j] = '#'

def remove_polyomino(grid, polyomino, x, y):
    for i in range(4):
        for j in range(4):
            if polyomino[i][j] == '#':
                grid[x + i][y + j] = '.'

def is_grid_filled(grid):
    for row in grid:
        if '.' in row:
            return False
    return True

def solve(grid, polyominoes, index):
    if index == 3:
        return is_grid_filled(grid)
    
    polyomino = polyominoes[index]
    rotations = get_all_rotations(polyomino)
    
    for rotation in rotations:
        for x in range(4):
            for y in range(4):
                if can_place(grid, rotation, x, y):
                    place_polyomino(grid, rotation, x, y)
                    if solve(grid, polyominoes, index + 1):
                        return True
                    remove_polyomino(grid, rotation, x, y)
    return False

def main():
    import sys
    input = sys.stdin.read().split()
    polyominoes = [input[i:i+4] for i in range(0, 12, 4)]
    
    grid = [['.' for _ in range(4)] for _ in range(4)]
    
    for perm in permutations(range(3)):
        if solve(grid, [polyominoes[i] for i in perm], 0):
            print("Yes")
            return
        grid = [['.' for _ in range(4)] for _ in range(4)]
    
    print("No")

if __name__ == "__main__":
    main()