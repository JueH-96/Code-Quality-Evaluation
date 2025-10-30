from collections import deque
import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(n):
        s = sys.stdin.readline().strip()
        grid.append([c == '.' for c in s])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    touched = [[False for _ in range(m)] for _ in range(n)]
    visited = [[False for _ in range(m)] for _ in range(n)]
    
    start_i, start_j = 1, 1
    queue = deque()
    queue.append((start_i, start_j))
    visited[start_i][start_j] = True
    
    while queue:
        i, j = queue.popleft()
        for dx, dy in directions:
            current_x, current_y = i, j
            path = [(current_x, current_y)]
            while True:
                next_x = current_x + dx
                next_y = current_y + dy
                if 0 <= next_x < n and 0 <= next_y < m and grid[next_x][next_y]:
                    current_x, current_y = next_x, next_y
                    path.append((current_x, current_y))
                else:
                    break
            for x, y in path:
                if not touched[x][y]:
                    touched[x][y] = True
            ex, ey = current_x, current_y
            if not visited[ex][ey]:
                visited[ex][ey] = True
                queue.append((ex, ey))
    
    count = 0
    for row in touched:
        count += sum(row)
    print(count)

if __name__ == "__main__":
    main()