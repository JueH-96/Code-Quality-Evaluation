def solve():
    N, M = map(int, input().split())
    grid = [input() for _ in range(N)]

    def dfs(i, j):
        if not (0 <= i < N) or not (0 <= j < M) or grid[i][j] == '#':
            return 0
        grid[i][j] = '#'
        return 1 + sum(dfs(x, y) for x, y in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)])

    print(dfs(2, 2))

solve()