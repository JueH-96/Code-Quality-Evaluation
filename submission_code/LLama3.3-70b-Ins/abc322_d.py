import sys
import itertools

def rotate(polyomino):
    return [list(reversed(i)) for i in zip(*polyomino)]

def get_polyominoes():
    polyominoes = []
    for _ in range(3):
        polyomino = [list(input()) for _ in range(4)]
        polyominoes.append(polyomino)
    return polyominoes

def get_rotations(polyomino):
    rotations = [polyomino]
    for _ in range(3):
        polyomino = rotate(polyomino)
        rotations.append(polyomino)
    return rotations

def get_all_polyominoes(polyominoes):
    all_polyominoes = []
    for polyomino in polyominoes:
        rotations = get_rotations(polyomino)
        all_polyominoes.extend(rotations)
    return all_polyominoes

def is_valid_position(grid, polyomino, x, y):
    for i in range(4):
        for j in range(4):
            if polyomino[i][j] == '#':
                if x + i < 0 or x + i >= 4 or y + j < 0 or y + j >= 4:
                    return False
                if grid[x + i][y + j] == '#':
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

def is_full(grid):
    for i in range(4):
        for j in range(4):
            if grid[i][j] == '.':
                return False
    return True

def solve(grid, polyominoes, index):
    if index == len(polyominoes):
        return is_full(grid)
    for x in range(4):
        for y in range(4):
            if is_valid_position(grid, polyominoes[index], x, y):
                place_polyomino(grid, polyominoes[index], x, y)
                if solve(grid, polyominoes, index + 1):
                    return True
                remove_polyomino(grid, polyominoes[index], x, y)
    return False

def main():
    polyominoes = get_polyominoes()
    all_polyominoes = get_all_polyominoes(polyominoes)
    grid = [['.' for _ in range(4)] for _ in range(4)]
    for p in itertools.permutations(all_polyominoes, 3):
        if solve(grid, list(p), 0):
            print('Yes')
            return
    print('No')

if __name__ == '__main__':
    main()