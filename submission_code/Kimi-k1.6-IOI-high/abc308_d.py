import sys
from collections import deque

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    if grid[0][0] != 's':
        print("No")
        return
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    target_i, target_j = H-1, W-1
    snuke = 'snuke'
    
    # 3D visited array: H rows, W cols, 5 cycle positions
    visited = [[[False]*5 for _ in range(W)] for __ in range(H)]
    queue = deque()
    queue.append((0, 0, 0))
    visited[0][0][0] = True
    found = False
    
    while queue:
        i, j, c = queue.popleft()
        if i == target_i and j == target_j:
            found = True
            break
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                next_c = (c + 1) % 5
                required_char = snuke[next_c]
                if grid[ni][nj] == required_char and not visited[ni][nj][next_c]:
                    visited[ni][nj][next_c] = True
                    queue.append((ni, nj, next_c))
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()