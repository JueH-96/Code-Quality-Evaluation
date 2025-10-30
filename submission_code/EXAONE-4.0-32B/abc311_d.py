from collections import deque
import sys

def main():
    data = sys.stdin.read().splitlines()
    n, m = map(int, data[0].split())
    grid = []
    for i in range(1, 1 + n):
        grid.append(data[i].strip())
    
    touched = [[False] * m for _ in range(n)]
    in_queue = [[False] * m for _ in range(n)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    start_r, start_c = 1, 1
    touched[start_r][start_c] = True
    in_queue[start_r][start_c] = True
    queue = deque()
    queue.append((start_r, start_c))
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr = r + dr
            nc = c + dc
            if grid[nr][nc] == '#':
                continue
                
            cr, cc = nr, nc
            while grid[cr + dr][cc + dc] == '.':
                cr += dr
                cc += dc
            
            if not touched[cr][cc]:
                touched[cr][cc] = True
            if not in_queue[cr][cc]:
                in_queue[cr][cc] = True
                queue.append((cr, cc))
    
    count = 0
    for i in range(n):
        for j in range(m):
            if touched[i][j]:
                count += 1
                
    print(count)

if __name__ == "__main__":
    main()