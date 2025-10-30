def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    H, W, K = map(int, input().split())
    grid = [input().strip() for _ in range(H)]
    
    # visited matrix to track used cells in the current path.
    visited = [[False] * W for _ in range(H)]
    ans = 0

    # DFS function that, starting from cell (i, j) having taken 'step' moves so far,
    # continues until we have taken K moves.
    # We manually check each adjacent cell (up, down, left, right) to reduce loop overhead.
    def dfs(i, j, step):
        nonlocal ans
        if step == K:
            ans += 1
            return
        ns = step + 1
        # Up
        ni = i - 1
        if ni >= 0:
            if grid[ni][j] == '.' and not visited[ni][j]:
                visited[ni][j] = True
                dfs(ni, j, ns)
                visited[ni][j] = False
        # Down
        ni = i + 1
        if ni < H:
            if grid[ni][j] == '.' and not visited[ni][j]:
                visited[ni][j] = True
                dfs(ni, j, ns)
                visited[ni][j] = False
        # Left
        nj = j - 1
        if nj >= 0:
            if grid[i][nj] == '.' and not visited[i][nj]:
                visited[i][nj] = True
                dfs(i, nj, ns)
                visited[i][nj] = False
        # Right
        nj = j + 1
        if nj < W:
            if grid[i][nj] == '.' and not visited[i][nj]:
                visited[i][nj] = True
                dfs(i, nj, ns)
                visited[i][nj] = False

    # Try every empty cell ('.') as a starting point.
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                visited[i][j] = True
                dfs(i, j, 0)
                visited[i][j] = False

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()