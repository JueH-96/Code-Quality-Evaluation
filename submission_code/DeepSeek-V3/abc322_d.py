# YOUR CODE HERE
import sys
from itertools import permutations

def read_polyomino():
    poly = []
    for _ in range(4):
        line = sys.stdin.readline().strip()
        poly.append(list(line))
    return poly

def get_shapes(poly):
    shapes = []
    for i in range(4):
        for j in range(4):
            if poly[i][j] == '#':
                shapes.append((i, j))
    return shapes

def rotate(shape):
    return [(y, -x) for x, y in shape]

def translate(shape, dx, dy):
    return [(x + dx, y + dy) for x, y in shape]

def is_valid(shape):
    for x, y in shape:
        if x < 0 or x >= 4 or y < 0 or y >= 4:
            return False
    return True

def can_fit(grid, shapes):
    for shape in shapes:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                translated = translate(shape, dx, dy)
                if is_valid(translated):
                    conflict = False
                    for x, y in translated:
                        if grid[x][y] != '.':
                            conflict = True
                            break
                    if not conflict:
                        for x, y in translated:
                            grid[x][y] = '#'
                        break
                else:
                    continue
            else:
                continue
            break
        else:
            return False
    return True

def main():
    poly1 = read_polyomino()
    poly2 = read_polyomino()
    poly3 = read_polyomino()
    
    shapes1 = get_shapes(poly1)
    shapes2 = get_shapes(poly2)
    shapes3 = get_shapes(poly3)
    
    # Generate all possible rotations for each polyomino
    rotations1 = [shapes1]
    for _ in range(3):
        shapes1 = rotate(shapes1)
        rotations1.append(shapes1)
    
    rotations2 = [shapes2]
    for _ in range(3):
        shapes2 = rotate(shapes2)
        rotations2.append(shapes2)
    
    rotations3 = [shapes3]
    for _ in range(3):
        shapes3 = rotate(shapes3)
        rotations3.append(shapes3)
    
    # Try all combinations of rotations
    for r1 in rotations1:
        for r2 in rotations2:
            for r3 in rotations3:
                grid = [['.' for _ in range(4)] for _ in range(4)]
                if can_fit(grid, [r1, r2, r3]):
                    # Check if the entire grid is filled
                    filled = True
                    for row in grid:
                        if '.' in row:
                            filled = False
                            break
                    if filled:
                        print("Yes")
                        return
    print("No")

if __name__ == "__main__":
    main()