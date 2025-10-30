from collections import defaultdict

def solve():
    H, W, Q = map(int, input().split())
    grid = [[True] * W for _ in range(H)]

    for _ in range(Q):
        r, c = map(int, input().split())
        r -= 1
        c -= 1

        if grid[r][c]:
            grid[r][c] = False
        else:
            # Look up
            for i in range(r-1, -1, -1):
                if grid[i][c]:
                    grid[i][c] = False
                    break
            # Look down
            for i in range(r+1, H):
                if grid[i][c]:
                    grid[i][c] = False
                    break
            # Look left
            for j in range(c-1, -1, -1):
                if grid[r][j]:
                    grid[r][j] = False
                    break
            # Look right
            for j in range(c+1, W):
                if grid[r][j]:
                    grid[r][j] = False
                    break

    return sum(sum(row) for row in grid)

print(solve())