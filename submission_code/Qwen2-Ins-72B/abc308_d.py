from collections import deque

def check_path(grid, h, w):
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    queue = deque([((0, 0), 0)])
    visited = set([(0, 0)])
    
    while queue:
        (x, y), step = queue.popleft()
        
        if x == h - 1 and y == w - 1 and step % 5 == 4:
            return True
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < h and 0 <= ny < w and (nx, ny) not in visited:
                if grid[nx][ny] == 'snuke'[step % 5]:
                    queue.append(((nx, ny), step + 1))
                    visited.add((nx, ny))
    
    return False

h, w = map(int, input().split())
grid = [input() for _ in range(h)]

print('Yes' if check_path(grid, h, w) else 'No')