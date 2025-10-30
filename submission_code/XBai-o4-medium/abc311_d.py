import sys
from collections import deque

def main():
    n, m = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(n)]
    
    touched = [[False]*m for _ in range(n)]
    processed = [[False]*m for _ in range(n)]
    
    count = 0
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    start_i, start_j = 1, 1
    processed[start_i][start_j] = True
    q = deque()
    q.append((start_i, start_j))
    
    while q:
        i, j = q.popleft()
        for dx, dy in directions:
            current_i, current_j = i, j
            path = [(current_i, current_j)]
            while True:
                next_i = current_i + dx
                next_j = current_j + dy
                if grid[next_i][next_j] == '#':
                    break
                current_i = next_i
                current_j = next_j
                path.append((current_i, current_j))
            # Update touched and count
            for x, y in path:
                if not touched[x][y]:
                    touched[x][y] = True
                    count += 1
            end_i, end_j = current_i, current_j
            if not processed[end_i][end_j]:
                processed[end_i][end_j] = True
                q.append((end_i, end_j))
    
    print(count)

if __name__ == "__main__":
    main()