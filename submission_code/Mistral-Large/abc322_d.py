import sys
from itertools import product

def rotate(polyomino):
    return [list(row) for row in zip(*polyomino[::-1])]

def can_place(grid, polyomino, x, y):
    for i in range(4):
        for j in range(4):
            if polyomino[i][j] == '#' and grid[x + i][y + j] != '.':
                return False
    return True

def place(grid, polyomino, x, y, place=True):
    for i in range(4):
        for j in range(4):
            if polyomino[i][j] == '#':
                grid[x + i][y + j] = '#' if place else '.'

def solve():
    input = sys.stdin.read().strip().split()
    polyominoes = []

    for i in range(3):
        polyomino = [list(input[j]) for j in range(i * 4, i * 4 + 4)]
        polyominoes.append(polyomino)

    grid = [['.' for _ in range(4)] for _ in range(4)]

    def backtrack(index):
        if index == 3:
            return all(cell == '#' for row in grid for cell in row)

        polyomino = polyominoes[index]
        for _ in range(4):
            polyomino = rotate(polyomino)
            for x, y in product(range(1), range(1)):
                if can_place(grid, polyomino, x, y):
                    place(grid, polyomino, x, y, place=True)
                    if backtrack(index + 1):
                        return True
                    place(grid, polyomino, x, y, place=False)
        return False

    if backtrack(0):
        print("Yes")
    else:
        print("No")

solve()