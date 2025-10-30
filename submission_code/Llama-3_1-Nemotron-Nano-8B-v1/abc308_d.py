from collections import deque
import sys

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    # Check starting cell
    if grid[0][0] != 's':
        print("No")
        return
    
    # Check ending cell's character is in snuke
    snuke = ['s', 'n', 'u', 'k', 'e']
    if grid[H-1][W-1] not in snuke:
        print("No")
        return
    
    # Directions: up, down, left, right
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Visited array: H x W x 5
    visited = [[[False]*5 for _ in range(W)] for _ in range(H)]
    
    q = deque()
    start_i, start_j = 0, 0
    start_m = 0  # since starting at step 1 (m=0)
    visited[start_i][start_j][start_m] = True
    q.append((start_i, start_j, start_m))
    
    found = False
    
    while q:
        i, j, m = q.popleft()
        
        # Check if current position is the end
        if i == H-1 and j == W-1:
            if grid[i][j] == snuke[m]:
                found = True
                break
        
        next_m = (m + 1) % 5
        next_char = snuke[next_m]
        
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == next_char and not visited[ni][nj][next_m]:
                    visited[ni][nj][next_m] = True
                    q.append((ni, nj, next_m))
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()