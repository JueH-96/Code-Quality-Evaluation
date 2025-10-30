import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(N):
        s = sys.stdin.readline().strip()
        grid.append(s)
    
    visited_ice = [[False for _ in range(M)] for _ in range(N)]
    visited_stop = [[False for _ in range(M)] for _ in range(N)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    queue = deque()
    start_i, start_j = 1, 1
    queue.append((start_i, start_j))
    visited_stop[start_i][start_j] = True
    visited_ice[start_i][start_j] = True
    
    while queue:
        i, j = queue.popleft()
        for di, dj in directions:
            current_x, current_y = i, j
            path = []
            while True:
                path.append((current_x, current_y))
                next_x = current_x + di
                next_y = current_y + dj
                if 0 <= next_x < N and 0 <= next_y < M and grid[next_x][next_y] == '.':
                    current_x, current_y = next_x, next_y
                else:
                    break
            # Mark all in path as visited_ice
            for (a, b) in path:
                if not visited_ice[a][b]:
                    visited_ice[a][b] = True
            # Check stopping position
            stop_x, stop_y = current_x, current_y
            if not visited_stop[stop_x][stop_y]:
                visited_stop[stop_x][stop_y] = True
                queue.append((stop_x, stop_y))
    
    answer = sum(sum(row) for row in visited_ice)
    print(answer)

if __name__ == "__main__":
    main()