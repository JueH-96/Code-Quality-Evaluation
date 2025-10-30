def solve():
    n, m = map(int, input().split())
    grid = [input() for _ in range(n)]
    
    visited = set()
    q = [(2, 2)]
    visited.add((2, 2))
    
    while q:
        r, c = q.pop(0)
        
        # Move up
        nr = r - 1
        while nr >= 0 and grid[nr][c] == '.':
            if (nr, c) not in visited:
                visited.add((nr, c))
                q.append((nr,c))
            nr -= 1
        
        # Move down
        nr = r + 1
        while nr < n and grid[nr][c] == '.':
            if (nr, c) not in visited:
                visited.add((nr, c))
                q.append((nr,c))
            nr += 1
            
        # Move left
        nc = c - 1
        while nc >= 0 and grid[r][nc] == '.':
            if (r, nc) not in visited:
                visited.add((r, nc))
                q.append((r,nc))
            nc -= 1
            
        # Move right
        nc = c + 1
        while nc < m and grid[r][nc] == '.':
            if (r, nc) not in visited:
                visited.add((r, nc))
                q.append((r,nc))
            nc += 1
            
    print(len(visited))

solve()