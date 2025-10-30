# YOUR CODE HERE

import sys

def read_input():
    polyominoes = []
    for _ in range(3):
        polyomino = []
        for _ in range(4):
            line = sys.stdin.readline().strip()
            polyomino.append(list(line))
        polyominoes.append(polyomino)
    return polyominoes

def rotate(polyomino):
    return [list(x) for x in zip(*polyomino[::-1])]

def flip(polyomino):
    return [x[::-1] for x in polyomino]

def is_valid(grid, polyomino, i, j):
    for x in range(4):
        for y in range(4):
            if polyomino[x][y] == '#' and grid[i+x][j+y] == '#':
                return False
    return True

def fill(grid, polyominoes, index):
    if index == len(polyominoes):
        return True
    polyomino = polyominoes[index]
    for _ in range(2):
        for _ in range(4):
            for i in range(7):
                for j in range(7):
                    if is_valid(grid, polyomino, i, j):
                        for x in range(4):
                            for y in range(4):
                                if polyomino[x][y] == '#':
                                    grid[i+x][j+y] = '#'
                        if fill(grid, polyominoes, index+1):
                            return True
                        for x in range(4):
                            for y in range(4):
                                if polyomino[x][y] == '#':
                                    grid[i+x][j+y] = '.'
            polyomino = rotate(polyomino)
        polyomino = flip(polyomino)
    return False

def solve():
    polyominoes = read_input()
    grid = [['.']*10 for _ in range(10)]
    if fill(grid, polyominoes, 0):
        print('Yes')
    else:
        print('No')

solve()