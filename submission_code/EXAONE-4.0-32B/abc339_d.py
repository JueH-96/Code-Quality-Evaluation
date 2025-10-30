from collections import deque

def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        print(-1)
        return
    
    N = int(data[0].strip())
    grid = []
    for i in range(1, 1+N):
        grid.append(data[i].strip())
    
    obstacle = [[False] * N for _ in range(N)]
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == '#':
                obstacle[i][j] = True
            elif grid[i][j] == 'P':
                players.append((i, j))
                
    if len(players) < 2:
        print(-1)
        return
        
    r1, c1 = players[0]
    r2, c2 = players[1]
    
    visited = [[[[False] * N for _ in range(N)] for _ in range(N)] for _ in range(N)]
    visited[r1][c1][r2][c2] = True
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    q = deque()
    q.append((r1, c1, r2, c2))
    steps = 0
    found = False
    
    while q:
        size = len(q)
        for _ in range(size):
            r1, c1, r2, c2 = q.popleft()
            
            if r1 == r2 and c1 == c2:
                print(steps)
                return
                
            for dr, dc in dirs:
                nr1 = r1 + dr
                nc1 = c1 + dc
                if not (0 <= nr1 < N and 0 <= nc1 < N) or obstacle[nr1][nc1]:
                    nr1, nc1 = r1, c1
                    
                nr2 = r2 + dr
                nc2 = c2 + dc
                if not (0 <= nr2 < N and 0 <= nc2 < N) or obstacle[nr2][nc2]:
                    nr2, nc2 = r2, c2
                    
                if not visited[nr1][nc1][nr2][nc2]:
                    visited[nr1][nc1][nr2][nc2] = True
                    q.append((nr1, nc1, nr2, nc2))
                    
        steps += 1
        
    print(-1)

if __name__ == '__main__':
    main()