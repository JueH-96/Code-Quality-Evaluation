import sys

def solve():
    H, W, Q = map(int, sys.stdin.readline().split())
    cells = [[1]*W for _ in range(H)]
    for _ in range(Q):
        R, C = map(int, sys.stdin.readline().split())
        if cells[R-1][C-1] == 1:
            cells[R-1][C-1] = 0
            for r in range(R):
                for c in range(C):
                    if cells[r][c] == 1:
                        cells[r][c] = 0
            for r in range(R, H):
                for c in range(C, W):
                    if cells[r][c] == 1:
                        cells[r][c] = 0
            for r in range(R, H):
                for c in range(C-1, -1, -1):
                    if cells[r][c] == 1:
                        cells[r][c] = 0
            for r in range(R-1, -1, -1):
                for c in range(C, W):
                    if cells[r][c] == 1:
                        cells[r][c] = 0
    return sum(sum(row) for row in cells)

print(solve())