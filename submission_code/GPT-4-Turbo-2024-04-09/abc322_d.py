def read_polyomino():
    return [input().strip() for _ in range(4)]

def count_hashes(poly):
    return sum(row.count('#') for row in poly)

def generate_rotations(poly):
    rotations = [poly]
    for _ in range(3):
        rotations.append([
            ''.join(rotations[-1][3 - col][row] for col in range(4))
            for row in range(4)
        ])
    return rotations

def fits_in_grid(grid, poly, row_offset, col_offset):
    for i in range(4):
        for j in range(4):
            if poly[i][j] == '#' and (i + row_offset >= 4 or j + col_offset >= 4 or grid[i + row_offset][j + col_offset] != '.'):
                return False
    return True

def place_in_grid(grid, poly, row_offset, col_offset):
    new_grid = [list(row) for row in grid]
    for i in range(4):
        for j in range(4):
            if poly[i][j] == '#':
                new_grid[i + row_offset][j + col_offset] = '#'
    return [''.join(row) for row in new_grid]

def remove_from_grid(grid, poly, row_offset, col_offset):
    new_grid = [list(row) for row in grid]
    for i in range(4):
        for j in range(4):
            if poly[i][j] == '#':
                new_grid[i + row_offset][j + col_offset] = '.'
    return [''.join(row) for row in new_grid]

def solve(grid, polys, index):
    if index == len(polys):
        return all(c == '#' for row in grid for c in row)
    
    poly = polys[index]
    rotations = generate_rotations(poly)
    
    for rotation in rotations:
        for row_offset in range(4):
            for col_offset in range(4):
                if fits_in_grid(grid, rotation, row_offset, col_offset):
                    new_grid = place_in_grid(grid, rotation, row_offset, col_offset)
                    if solve(new_grid, polys, index + 1):
                        return True
                    grid = remove_from_grid(grid, rotation, row_offset, col_offset)
    
    return False

def main():
    poly1 = read_polyomino()
    poly2 = read_polyomino()
    poly3 = read_polyomino()
    
    polys = [poly1, poly2, poly3]
    grid = ['....', '....', '....', '....']
    
    # Check if the total number of '#' matches 16 (4x4 grid)
    if sum(count_hashes(poly) for poly in polys) != 16:
        print("No")
        return
    
    if solve(grid, polys, 0):
        print("Yes")
    else:
        print("No")

main()