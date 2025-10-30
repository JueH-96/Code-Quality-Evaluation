import sys
from collections import deque

def main():
    n, m = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(n)]
    
    touched = [[False] * m for _ in range(n)]
    visited = [[False] * m for _ in range(n)]
    
    start_i, start_j = 1, 1  # 0-based index for the starting position (2,2) in 1-based
    touched[start_i][start_j] = True
    visited[start_i][start_j] = True
    queue = deque([(start_i, start_j)])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    while queue:
        i, j = queue.popleft()
        for dr, dc in directions:
            path = []
            current_i, current_j = i + dr, j + dc
            while 0 <= current_i < n and 0 <= current_j < m and grid[current_i][current_j] == '.':
                path.append((current_i, current_j))
                current_i += dr
                current_j += dc
            # Mark all squares in the path as touched
            for x, y in path:
                if not touched[x][y]:
                    touched[x][y] = True
            # Check if the path is non-empty and process the new resting position
            if path:
                new_i, new_j = path[-1]
                if not visited[new_i][new_j]:
                    visited[new_i][new_j] = True
                    queue.append((new_i, new_j))
    
    # Count all touched squares
    print(sum(row.count(True) for row in touched))

if __name__ == "__main__":
    main()