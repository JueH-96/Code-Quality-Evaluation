import sys
from collections import deque

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    if grid[0][0] != 's':
        print("No")
        return
    
    chars = ['s', 'n', 'u', 'k', 'e']
    target = (H-1, W-1)
    
    visited = [[[False]*5 for _ in range(W)] for _ in range(H)]
    q = deque()
    q.append((0, 0, 0))
    visited[0][0][0] = True
    
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    while q:
        i, j, m = q.popleft()
        if (i, j) == target:
            print("Yes")
            return
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                next_m = (m + 1) % 5
                if grid[ni][nj] == chars[next_m]:
                    if not visited[ni][nj][next_m]:
                        visited[ni][nj][next_m] = True
                        q.append((ni, nj, next_m))
    
    print("No")

if __name__ == "__main__":
    main()