def solve():
    n, m = map(int, input().split())
    grid = [input() for _ in range(n)]
    visited = [[False] * m for _ in range(n)]
    q = [(1, 1)]
    visited[1][1] = True
    
    while q:
        r, c = q.pop(0)
        
        # Move up
        nr = r - 1
        while grid[nr][c] == '.':
            visited[nr][c] = True
            nr -= 1
        if nr + 1 != r and not visited[nr+1][c]:
            q.append((nr + 1, c))
            
        # Move down
        nr = r + 1
        while grid[nr][c] == '.':
            visited[nr][c] = True
            nr += 1
        if nr - 1 != r and not visited[nr-1][c]:
            q.append((nr - 1, c))
            
        # Move left
        nc = c - 1
        while grid[r][nc] == '.':
            visited[r][nc] = True
            nc -= 1
        if nc + 1 != c and not visited[r][nc+1]:
            q.append((r, nc + 1))
            
        # Move right
        nc = c + 1
        while grid[r][nc] == '.':
            visited[r][nc] = True
            nc += 1
        if nc - 1 != c and not visited[r][nc-1]:
            q.append((r, nc - 1))

    count = 0
    for i in range(n):
        for j in range(m):
            if visited[i][j]:
                count += 1
    print(count)

solve()