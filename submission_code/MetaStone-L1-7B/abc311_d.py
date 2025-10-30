import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(N):
        s = sys.stdin.readline().strip()
        grid.append(s)
    
    visited = [[False for _ in range(M)] for __ in range(N)]
    processed = [[False for _ in range(M)] for __ in range(N)]
    directions = [ (-1,0), (1,0), (0,-1), (0,1) ]
    
    q = deque()
    start_i = 1
    start_j = 1
    visited[start_i][start_j] = True
    processed[start_i][start_j] = True
    q.append( (start_i, start_j) )
    
    while q:
        i, j = q.popleft()
        for di, dj in directions:
            path = []
            x, y = i, j
            path.append( (x, y) )
            while True:
                x += di
                y += dj
                if x < 0 or x >= N or y <0 or y >= M:
                    break
                if grid[x][y] == '#':
                    break
                path.append( (x, y) )
            # Mark all in path as visited
            for x, y in path:
                if not visited[x][y]:
                    visited[x][y] = True
            # Find stopping square
            stop_i, stop_j = path[-1]
            if not processed[stop_i][stop_j]:
                processed[stop_i][stop_j] = True
                q.append( (stop_i, stop_j) )
    
    # Count the visited squares
    count = 0
    for i in range(N):
        for j in range(M):
            if visited[i][j]:
                count +=1
    print(count)

if __name__ == '__main__':
    main()