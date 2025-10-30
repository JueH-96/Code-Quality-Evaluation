def solve():
    polyominoes = []
    for _ in range(3):
        polyomino = []
        for _ in range(4):
            polyomino.append(input())
        polyominoes.append(polyomino)

    def rotate(p):
        return ["".join(p[j][i] for j in range(3,-1,-1)) for i in range(4)]

    def check(grid, p, r, c):
        for i in range(4):
            for j in range(4):
                if p[i][j] == '#' and (r + i >= 4 or c + j >= 4 or grid[r + i][c + j] != '.'):
                    return False
        for i in range(4):
            for j in range(4):
                if p[i][j] == '#':
                    grid[r + i][c + j] = '#'
        return True

    def backtrack(grid, poly_idx):
        if poly_idx == 3:
            for row in grid:
                if '.' in row:
                    return False
            return True

        p = polyominoes[poly_idx]
        for rot in range(4):
            for r in range(4):
                for c in range(4):
                    new_grid = [list(row) for row in grid]
                    if check(new_grid, p, r, c):
                        if backtrack(new_grid, poly_idx + 1):
                            return True
            p = rotate(p)
        return False

    grid = [['.' for _ in range(4)] for _ in range(4)]
    if backtrack(grid, 0):
        print("Yes")
    else:
        print("No")

solve()