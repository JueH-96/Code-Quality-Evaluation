def can_fill_grid(polyominoes):
    grid = [['.' for _ in range(4)] for _ in range(4)]
    
    def rotate(shape):
        return [''.join(shape[3-j][i] for j in range(4)) for i in range(4)]

    def get_shapes(polyomino):
        shapes = []
        current_shape = polyomino
        for _ in range(4):
            shapes.append(current_shape)
            current_shape = rotate(current_shape)
        return shapes

    def can_place(shape, r, c):
        for i in range(4):
            for j in range(4):
                if shape[i][j] == '#' and (r + i >= 4 or c + j >= 4 or grid[r + i][c + j] == '#'):
                    return False
        return True

    def place(shape, r, c):
        for i in range(4):
            for j in range(4):
                if shape[i][j] == '#':
                    grid[r + i][c + j] = '#'

    def remove(shape, r, c):
        for i in range(4):
            for j in range(4):
                if shape[i][j] == '#':
                    grid[r + i][c + j] = '.'

    def backtrack(index):
        if index == 3:
            return all(grid[i][j] == '#' for i in range(4) for j in range(4))

        for shape in get_shapes(polyominoes[index]):
            for r in range(4):
                for c in range(4):
                    if can_place(shape, r, c):
                        place(shape, r, c)
                        if backtrack(index + 1):
                            return True
                        remove(shape, r, c)
        return False

    return backtrack(0)

import sys
input = sys.stdin.read
data = input().strip().splitlines()

polyominoes = []
for i in range(0, 12, 4):
    polyomino = data[i:i+4]
    polyominoes.append(polyomino)

if can_fill_grid(polyominoes):
    print("Yes")
else:
    print("No")