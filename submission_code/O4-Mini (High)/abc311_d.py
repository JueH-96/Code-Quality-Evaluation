import sys
from collections import deque

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    grid = [list(input().rstrip()) for _ in range(N)]
    
    # visitedRest[i][j] = whether we've enqueued (i,j) as a resting position
    visitedRest = [[False]*M for _ in range(N)]
    # touched[i][j] = whether we've ever passed through or stopped on (i,j)
    touched = [[False]*M for _ in range(N)]
    
    # BFS on resting positions
    dq = deque()
    start = (1, 1)  # zero‐indexed coordinates for (2,2) in problem statement
    dq.append(start)
    visitedRest[1][1] = True
    touched[1][1] = True
    
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    while dq:
        i, j = dq.popleft()
        # try sliding in each of the 4 directions
        for di, dj in dirs:
            ni, nj = i, j
            # slide until next cell is rock
            while True:
                ti, tj = ni + di, nj + dj
                if grid[ti][tj] == '#':
                    break
                ni, nj = ti, tj
                if not touched[ni][nj]:
                    touched[ni][nj] = True
            # ni,nj is the resting cell after the slide
            if (ni, nj) != (i, j) and not visitedRest[ni][nj]:
                visitedRest[ni][nj] = True
                dq.append((ni, nj))
    
    # count all touched ice cells
    ans = 0
    for i in range(N):
        for j in range(M):
            if touched[i][j] and grid[i][j] == '.':
                ans += 1
    
    print(ans)

if __name__ == '__main__':
    main()