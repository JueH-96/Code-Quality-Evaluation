# YOUR CODE HERE
import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    N, M = map(int, input[:2])
    grid = input[2:2+N]
    
    touched = [[False]*M for _ in range(N)]
    visited = [[False]*M for _ in range(N)]
    
    start_i, start_j = 1, 1  # 0-based indexing for (2,2)
    queue = deque()
    queue.append((start_i, start_j))
    visited[start_i][start_j] = True
    touched[start_i][start_j] = True
    
    directions = [(-1,0), (1,0), (0,-1), (0,1)]  # up, down, left, right
    
    while queue:
        i, j = queue.popleft()
        for di, dj in directions:
            ni, nj = i, j
            while True:
                ti, tj = ni + di, nj + dj
                if 0 <= ti < N and 0 <= tj < M and grid[ti][tj] == '.':
                    ni, nj = ti, tj
                    if not touched[ni][nj]:
                        touched[ni][nj] = True
                else:
                    break
            if not visited[ni][nj]:
                visited[ni][nj] = True
                queue.append((ni, nj))
    
    count = sum(row.count(True) for row in touched)
    print(count)

if __name__ == "__main__":
    main()