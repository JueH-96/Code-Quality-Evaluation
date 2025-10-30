from collections import deque
import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    
    H = int(data[index])
    W = int(data[index + 1])
    index += 2
    
    F = []
    for _ in range(H):
        F.append(list(map(int, data[index:index + W])))
        index += W
    
    Q = int(data[index])
    index += 1
    
    queries = []
    for _ in range(Q):
        A = int(data[index]) - 1
        B = int(data[index + 1]) - 1
        Y = int(data[index + 2])
        C = int(data[index + 3]) - 1
        D = int(data[index + 4]) - 1
        Z = int(data[index + 5])
        queries.append((A, B, Y, C, D, Z))
        index += 6
    
    results = []
    
    for A, B, Y, C, D, Z in queries:
        queue = deque([(A, B, Y, 0)])
        visited = set()
        visited.add((A, B, Y))
        
        while queue:
            x, y, floor, stairs = queue.popleft()
            
            if (x, y, floor) == (C, D, Z):
                results.append(stairs)
                break
            
            # Move within the same building
            if floor > 1 and (x, y, floor - 1) not in visited:
                visited.add((x, y, floor - 1))
                queue.append((x, y, floor - 1, stairs + 1))
            
            if floor < F[x][y] and (x, y, floor + 1) not in visited:
                visited.add((x, y, floor + 1))
                queue.append((x, y, floor + 1, stairs + 1))
            
            # Move to adjacent buildings
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and F[nx][ny] >= floor and (nx, ny, floor) not in visited:
                    visited.add((nx, ny, floor))
                    queue.append((nx, ny, floor, stairs))
    
    for result in results:
        print(result)