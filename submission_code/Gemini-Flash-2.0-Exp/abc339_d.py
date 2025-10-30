from collections import deque

def solve():
    n = int(input())
    grid = [input() for _ in range(n)]
    
    players = []
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 'P':
                players.append((r, c))
    
    r1, c1 = players[0]
    r2, c2 = players[1]
    
    q = deque([(r1, c1, r2, c2, 0)])
    visited = set()
    visited.add((r1, c1, r2, c2))
    
    while q:
        curr_r1, curr_c1, curr_r2, curr_c2, moves = q.popleft()
        
        if (curr_r1, curr_c1) == (curr_r2, curr_c2):
            print(moves)
            return
        
        # Move up
        new_r1 = curr_r1 - 1 if curr_r1 > 0 and grid[curr_r1 - 1][curr_c1] != '#' else curr_r1
        new_r2 = curr_r2 - 1 if curr_r2 > 0 and grid[curr_r2 - 1][curr_c2] != '#' else curr_r2
        
        if (new_r1, curr_c1, new_r2, curr_c2) not in visited:
            q.append((new_r1, curr_c1, new_r2, curr_c2, moves + 1))
            visited.add((new_r1, curr_c1, new_r2, curr_c2))
            
        # Move down
        new_r1 = curr_r1 + 1 if curr_r1 < n - 1 and grid[curr_r1 + 1][curr_c1] != '#' else curr_r1
        new_r2 = curr_r2 + 1 if curr_r2 < n - 1 and grid[curr_r2 + 1][curr_c2] != '#' else curr_r2
        
        if (new_r1, curr_c1, new_r2, curr_c2) not in visited:
            q.append((new_r1, curr_c1, new_r2, curr_c2, moves + 1))
            visited.add((new_r1, curr_c1, new_r2, curr_c2))
            
        # Move left
        new_c1 = curr_c1 - 1 if curr_c1 > 0 and grid[curr_r1][curr_c1 - 1] != '#' else curr_c1
        new_c2 = curr_c2 - 1 if curr_c2 > 0 and grid[curr_r2][curr_c2 - 1] != '#' else curr_c2
        
        if (curr_r1, new_c1, curr_r2, new_c2) not in visited:
            q.append((curr_r1, new_c1, curr_r2, new_c2, moves + 1))
            visited.add((curr_r1, new_c1, curr_r2, new_c2))
            
        # Move right
        new_c1 = curr_c1 + 1 if curr_c1 < n - 1 and grid[curr_r1][curr_c1 + 1] != '#' else curr_c1
        new_c2 = curr_c2 + 1 if curr_c2 < n - 1 and grid[curr_r2][curr_c2 + 1] != '#' else curr_c2
        
        if (curr_r1, new_c1, curr_r2, new_c2) not in visited:
            q.append((curr_r1, new_c1, curr_r2, new_c2, moves + 1))
            visited.add((curr_r1, new_c1, curr_r2, new_c2))
            
    print(-1)

solve()