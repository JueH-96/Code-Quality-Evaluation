from collections import deque

def bfs(grid, n, start1, start2):
    queue = deque([(start1, start2, 0)])
    visited = set([(start1, start2)])
    
    while queue:
        (x1, y1), (x2, y2), steps = queue.popleft()
        
        if (x1, y1) == (x2, y2):
            return steps
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx1, ny1 = x1 + dx, y1 + dy
            nx2, ny2 = x2 + dx, y2 + dy
            
            if 0 <= nx1 < n and 0 <= ny1 < n and grid[nx1][ny1] != '#' and (nx1, ny1) != (x2, y2):
                if (nx1, ny1, x2, y2) not in visited:
                    visited.add((nx1, ny1, x2, y2))
                    queue.append(((nx1, ny1), (x2, y2), steps + 1))
            
            if 0 <= nx2 < n and 0 <= ny2 < n and grid[nx2][ny2] != '#' and (nx2, ny2) != (x1, y1):
                if (x1, y1, nx2, ny2) not in visited:
                    visited.add((x1, y1, nx2, ny2))
                    queue.append(((x1, y1), (nx2, ny2), steps + 1))
    
    return -1

def main():
    n = int(input())
    grid = [list(input()) for _ in range(n)]
    
    start1, start2 = None, None
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                if start1 is None:
                    start1 = (i, j)
                else:
                    start2 = (i, j)
    
    print(bfs(grid, n, start1, start2))

if __name__ == "__main__":
    main()