import sys

def read_input():
    H, W, Y = map(int, sys.stdin.readline().split())
    A = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]
    return H, W, Y, A

def solve(H, W, Y, A):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W

    def dfs(x, y, sea_level, visited):
        if not is_valid(x, y) or A[x][y] <= sea_level or visited[x][y]:
            return
        visited[x][y] = True
        for dx, dy in directions:
            dfs(x + dx, y + dy, sea_level, visited)

    for year in range(1, Y + 1):
        sea_level = year
        visited = [[False] * W for _ in range(H)]
        for i in range(H):
            for j in range(W):
                if A[i][j] <= sea_level:
                    visited[i][j] = True
        for i in range(H):
            for j in range(W):
                if visited[i][j]:
                    for dx, dy in directions:
                        dfs(i + dx, y + dy, sea_level, visited)
        area = sum(1 for i in range(H) for j in range(W) if not visited[i][j])
        print(area)

H, W, Y, A = read_input()
solve(H, W, Y, A)