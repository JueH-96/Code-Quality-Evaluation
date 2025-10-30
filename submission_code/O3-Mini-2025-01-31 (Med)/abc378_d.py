def main():
    import sys
    sys.setrecursionlimit(10000)
    input = sys.stdin.readline

    H, W, K = map(int, input().split())
    grid = [list(input().strip()) for _ in range(H)]
    
    # Directions: up, right, down, left
    direction = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    ans = 0
    
    def dfs(i, j, moves, visited):
        nonlocal ans
        # If we made exactly K moves, then we have a valid sequence (length K+1)
        if moves == K:
            ans += 1
            return
        
        for di, dj in direction:
            ni, nj = i + di, j + dj
            # Check boundaries, cell emptiness and if not visited.
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.' and not visited[ni][nj]:
                visited[ni][nj] = True
                dfs(ni, nj, moves + 1, visited)
                visited[ni][nj] = False

    # For every empty cell as starting point, perform DFS search.
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':  # starting cell must be empty
                visited = [[False] * W for _ in range(H)]
                visited[i][j] = True
                dfs(i, j, 0, visited)

    print(ans)


if __name__ == '__main__':
    main()