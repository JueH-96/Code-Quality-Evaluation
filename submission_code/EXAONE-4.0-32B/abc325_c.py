from collections import deque
import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
        
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1 + H):
        grid.append(data[i].strip())
    
    if H == 0 or W == 0:
        print(0)
        return
        
    directions = [ (dx, dy) for dx in (-1,0,1) for dy in (-1,0,1) if not (dx == 0 and dy == 0) ]
    visited = [[False] * W for _ in range(H)]
    count = 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                count += 1
                queue = deque()
                queue.append((i, j))
                visited[i][j] = True
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if not visited[nx][ny] and grid[nx][ny] == '#':
                                visited[nx][ny] = True
                                queue.append((nx, ny))
                                
    print(count)

if __name__ == '__main__':
    main()