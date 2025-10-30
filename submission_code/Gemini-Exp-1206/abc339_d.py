from collections import deque

def solve():
    n = int(input())
    grid = [input() for _ in range(n)]
    
    players = []
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 'P':
                players.append((r, c))
    
    q = deque([(players[0][0], players[0][1], players[1][0], players[1][1], 0)])
    visited = set()
    visited.add((players[0][0], players[0][1], players[1][0], players[1][1]))
    
    while q:
        r1, c1, r2, c2, moves = q.popleft()
        
        if (r1, c1) == (r2, c2):
            print(moves)
            return
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr1, nc1 = r1 + dr, c1 + dc
            nr2, nc2 = r2 + dr, c2 + dc
            
            if 0 <= nr1 < n and 0 <= nc1 < n and grid[nr1][nc1] != '#':
                pass
            else:
                nr1, nc1 = r1, c1
                
            if 0 <= nr2 < n and 0 <= nc2 < n and grid[nr2][nc2] != '#':
                pass
            else:
                nr2, nc2 = r2, c2
            
            if (nr1, nc1, nr2, nc2) not in visited:
                visited.add((nr1, nc1, nr2, nc2))
                q.append((nr1, nc1, nr2, nc2, moves + 1))
    
    print(-1)

solve()