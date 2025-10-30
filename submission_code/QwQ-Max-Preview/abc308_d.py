from collections import deque
import sys

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    if grid[0][0] != 's':
        print("No")
        return
    
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    snuke = 'snuke'
    visited = [[[False] * 5 for _ in range(W)] for _ in range(H)]
    queue = deque()
    queue.append((0, 0, 0))
    visited[0][0][0] = True
    
    while queue:
        i, j, k = queue.popleft()
        if i == H - 1 and j == W - 1:
            print("Yes")
            return
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                next_k = (k + 1) % 5
                if grid[ni][nj] == snuke[next_k] and not visited[ni][nj][next_k]:
                    visited[ni][nj][next_k] = True
                    queue.append((ni, nj, next_k))
    
    print("No")

if __name__ == "__main__":
    main()