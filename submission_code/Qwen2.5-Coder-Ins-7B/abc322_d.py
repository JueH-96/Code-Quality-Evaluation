import sys

def can_place(polyomino, grid, x, y):
    for i in range(4):
        for j in range(4):
            if polyomino[i][j] == '#' and (x + i >= 4 or y + j >= 4 or grid[x + i][y + j] != '.'):
                return False
    return True

def place(polyomino, grid, x, y):
    for i in range(4):
        for j in range(4):
            if polyomino[i][j] == '#':
                grid[x + i][y + j] = '#'

def remove(polyomino, grid, x, y):
    for i in range(4):
        for j in range(4):
            if polyomino[i][j] == '#':
                grid[x + i][y + j] = '.'

def solve(polyominos, grid, placed):
    if len(placed) == 3:
        return True
    for i in range(4):
        for j in range(4):
            if grid[i][j] == '.':
                for polyomino in polyominos:
                    if can_place(polyomino, grid, i, j):
                        place(polyomino, grid, i, j)
                        placed.append(polyomino)
                        if solve(polyominos, grid, placed):
                            return True
                        remove(polyomino, grid, i, j)
                        placed.pop()
    return False

def main():
    polyominos = []
    grid = []
    for _ in range(12):
        line = sys.stdin.readline().strip()
        polyominos.append([line[i:i+4] for i in range(0, 16, 4)])
    for _ in range(4):
        grid.append(list(sys.stdin.readline().strip()))
    if solve(polyominos, grid, []):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()