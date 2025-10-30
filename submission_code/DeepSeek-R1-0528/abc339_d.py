import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print(-1)
        return
    N = int(data[0].strip())
    grid = []
    for i in range(1, 1+N):
        grid.append(data[i].strip())
    
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append((i, j))
                
    if len(players) < 2:
        print(-1)
        return
        
    r1, c1 = players[0]
    r2, c2 = players[1]
    
    dist = [[[[-1 for _ in range(N)] for __ in range(N)] for ___ in range(N)] for ____ in range(N)]
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    q = deque()
    dist[r1][c1][r2][c2] = 0
    q.append((r1, c1, r2, c2))
    
    while q:
        ra, ca, rb, cb = q.popleft()
        current_moves = dist[ra][ca][rb][cb]
        
        for dr, dc in directions:
            nra = ra + dr
            nca = ca + dc
            if nra < 0 or nra >= N or nca < 0 or nca >= N or grid[nra][nca] == '#':
                nra, nca = ra, ca
                
            nrb = rb + dr
            ncb = cb + dc
            if nrb < 0 or nrb >= N or ncb < 0 or ncb >= N or grid[nrb][ncb] == '#':
                nrb, ncb = rb, cb
                
            if nra == nrb and nca == ncb:
                print(current_moves + 1)
                return
                
            if dist[nra][nca][nrb][ncb] == -1:
                dist[nra][nca][nrb][ncb] = current_moves + 1
                q.append((nra, nca, nrb, ncb))
                
    print(-1)

if __name__ == "__main__":
    main()