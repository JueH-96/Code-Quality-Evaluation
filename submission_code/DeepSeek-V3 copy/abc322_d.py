# YOUR CODE HERE
import sys
from itertools import permutations

def read_polyomino():
    poly = []
    for _ in range(4):
        line = sys.stdin.readline().strip()
        poly.append(list(line))
    return poly

def get_cells(poly):
    cells = []
    for i in range(4):
        for j in range(4):
            if poly[i][j] == '#':
                cells.append((i, j))
    return cells

def translate(cells, dx, dy):
    return [(x + dx, y + dy) for x, y in cells]

def rotate(cells):
    return [(y, 3 - x) for x, y in cells]

def is_valid(cells):
    for x, y in cells:
        if x < 0 or x >= 4 or y < 0 or y >= 4:
            return False
    return True

def can_fit(grid, cells):
    for x, y in cells:
        if grid[x][y] != '.':
            return False
    return True

def place_polyomino(grid, cells):
    for x, y in cells:
        grid[x][y] = '#'

def remove_polyomino(grid, cells):
    for x, y in cells:
        grid[x][y] = '.'

def solve():
    polys = []
    for _ in range(3):
        polys.append(read_polyomino())
    
    all_cells = []
    for poly in polys:
        cells = get_cells(poly)
        all_cells.append(cells)
    
    # Generate all possible rotations and translations for each polyomino
    possible = []
    for cells in all_cells:
        variants = set()
        for _ in range(4):
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    translated = translate(cells, dx, dy)
                    if is_valid(translated):
                        variants.add(tuple(sorted(translated)))
            cells = rotate(cells)
        possible.append(list(variants))
    
    # Try all permutations of the polyominoes
    for perm in permutations(range(3)):
        # Try all combinations of variants
        for var1 in possible[perm[0]]:
            grid = [['.' for _ in range(4)] for _ in range(4)]
            if not can_fit(grid, var1):
                continue
            place_polyomino(grid, var1)
            for var2 in possible[perm[1]]:
                if not can_fit(grid, var2):
                    continue
                place_polyomino(grid, var2)
                for var3 in possible[perm[2]]:
                    if not can_fit(grid, var3):
                        continue
                    place_polyomino(grid, var3)
                    # Check if the grid is fully covered
                    full = True
                    for i in range(4):
                        for j in range(4):
                            if grid[i][j] == '.':
                                full = False
                                break
                        if not full:
                            break
                    if full:
                        print("Yes")
                        return
                    remove_polyomino(grid, var3)
                remove_polyomino(grid, var2)
            remove_polyomino(grid, var1)
    print("No")

if __name__ == "__main__":
    solve()