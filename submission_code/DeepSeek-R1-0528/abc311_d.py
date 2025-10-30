import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        print(0)
        return
    n, m = map(int, data[0].split())
    grid = []
    for i in range(1, 1+n):
        grid.append(data[i].strip())
    
    passed = set()
    visited_rest = set()
    start = (1, 1)
    passed.add(start)
    visited_rest.add(start)
    q = deque([start])
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            path = []
            step = 1
            while True:
                nr = r + step * dr
                nc = c + step * dc
                if nr < 0 or nr >= n or nc < 0 or nc >= m:
                    break
                if grid[nr][nc] == '#':
                    break
                path.append((nr, nc))
                step += 1
                
            if path:
                new_stop = path[-1]
                passed.update(path)
                
                if new_stop not in visited_rest:
                    visited_rest.add(new_stop)
                    q.append(new_stop)
                    
    print(len(passed))

if __name__ == "__main__":
    main()