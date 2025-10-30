# YOUR CODE HERE
import sys

def solve():
    h, w = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(h)]
    
    sensors = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '#':
                sensors.append((r, c))
    
    if not sensors:
        print(0)
        return

    visited = [False] * len(sensors)
    count = 0
    
    for i in range(len(sensors)):
        if not visited[i]:
            count += 1
            q = [i]
            visited[i] = True
            
            while q:
                curr = q.pop(0)
                r1, c1 = sensors[curr]
                
                for j in range(len(sensors)):
                    if not visited[j]:
                        r2, c2 = sensors[j]
                        if max(abs(r1 - r2), abs(c1 - c2)) == 1:
                            visited[j] = True
                            q.append(j)
    print(count)

solve()