import sys

def solve():
    N = int(input())
    grid = [list(input()) for _ in range(N)]

    for i in range(1, N//2 + 1):
        for x in range(i, N + 1 - i):
            for y in range(i, N + 1 - i):
                grid[y-1][N-x] = grid[x-1][y-1]

    for row in grid:
        print(''.join(row))

if __name__ == '__main__':
    solve()