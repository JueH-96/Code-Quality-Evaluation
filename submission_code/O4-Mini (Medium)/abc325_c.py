import sys
from collections import deque

def main():
    input = sys.stdin.readline
    H, W = map(int, input().split())
    grid = [list(input().rstrip('
')) for _ in range(H)]
    
    # 8 possible directions (vertical, horizontal, diagonal)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]
    
    def bfs(start_i, start_j):
        """Mark all cells in the same connected component as visited."""
        queue = deque()
        queue.append((start_i, start_j))
        grid[start_i][start_j] = '.'  # mark as visited immediately
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#':
                    grid[nx][ny] = '.'  # mark visited
                    queue.append((nx, ny))
    
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                # Found an unvisited sensor component
                count += 1
                bfs(i, j)
    
    print(count)

if __name__ == "__main__":
    main()