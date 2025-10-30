from collections import deque
import sys

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    if grid[0][0] != 's':
        print("No")
        return
    
    snuke = 'snuke'
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    
    visited = [[[False] * 5 for _ in range(W)] for _ in range(H)]
    queue = deque()
    queue.append((0, 0, 0))
    visited[0][0][0] = True
    
    found = False
    
    while queue:
        i, j, m = queue.popleft()
        
        if i == H-1 and j == W-1:
            found = True
            break
        
        for di, dj in dirs:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                next_m = (m + 1) % 5
                if grid[ni][nj] == snuke[next_m]:
                    if not visited[ni][nj][next_m]:
                        visited[ni][nj][next_m] = True
                        queue.append((ni, nj, next_m))
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()