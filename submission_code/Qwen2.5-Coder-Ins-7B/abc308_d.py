# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
grid = data[2:]

def is_valid_path():
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = [(0, 0, 0)]
    visited = set()
    
    while queue:
        x, y, step = queue.pop(0)
        if (x, y, step) in visited:
            continue
        visited.add((x, y, step))
        
        if step == 5 * H * W - 1:
            return True
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in visited:
                if grid[nx][ny] == 's':
                    queue.append((nx, ny, step + 1))
                elif grid[nx][ny] == 'n':
                    queue.append((nx, ny, step + 1))
                elif grid[nx][ny] == 'u':
                    queue.append((nx, ny, step + 1))
                elif grid[nx][ny] == 'k':
                    queue.append((nx, ny, step + 1))
                elif grid[nx][ny] == 'e':
                    queue.append((nx, ny, step + 1))
    
    return False

if is_valid_path():
    print("Yes")
else:
    print("No")