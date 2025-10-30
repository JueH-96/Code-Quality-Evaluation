# YOUR CODE HERE
def rotate_90(matrix):
    return list(zip(*matrix[::-1]))

def get_all_rotations(polyomino):
    rotations = [polyomino]
    for _ in range(3):
        polyomino = rotate_90(polyomino)
        rotations.append(polyomino)
    return rotations

def can_place(grid, polyomino, start_row, start_col):
    for r in range(4):
        for c in range(4):
            if polyomino[r][c] == '#':
                grid_r = start_row + r
                grid_c = start_col + c
                if grid_r >= 4 or grid_c >= 4 or grid[grid_r][grid_c] != '.':
                    return False
    return True

def place_polyomino(grid, polyomino, start_row, start_col, char):
    for r in range(4):
        for c in range(4):
            if polyomino[r][c] == '#':
                grid[start_row + r][start_col + c] = char

def backtrack(grid, polyominoes, index):
    if index == 3:
        return all(grid[r][c] != '.' for r in range(4) for c in range(4))
    
    for rotation in get_all_rotations(polyominoes[index]):
        for start_row in range(4):
            for start_col in range(4):
                if can_place(grid, rotation, start_row, start_col):
                    place_polyomino(grid, rotation, start_row, start_col, str(index))
                    if backtrack(grid, polyominoes, index + 1):
                        return True
                    place_polyomino(grid, rotation, start_row, start_col, '.')
    return False

def solve():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    polyominoes = []
    for i in range(3):
        polyomino = [list(data[i * 4 + j]) for j in range(4)]
        polyominoes.append(polyomino)
    
    grid = [['.' for _ in range(4)] for _ in range(4)]
    
    if backtrack(grid, polyominoes, 0):
        print("Yes")
    else:
        print("No")