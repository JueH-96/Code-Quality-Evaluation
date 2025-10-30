import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N, M = int(data[0]), int(data[1])
    grid = data[2:N+2]
    
    visited = [[False] * M for _ in range(N)]
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    def dfs(x, y):
        if visited[x][y]:
            return 0
        visited[x][y] = True
        count = 1
        for dx, dy in directions:
            nx, ny = x, y
            while 0 <= nx + dx < N and 0 <= ny + dy < M and grid[nx + dx][ny + dy] == '.':
                nx += dx
                ny += dy
            if not visited[nx][ny]:
                count += dfs(nx, ny)
        return count
    
    print(dfs(1, 1))

solve()