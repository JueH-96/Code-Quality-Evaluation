import sys
from collections import deque

def main():
    H, W, D = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    # Collect all H positions
    h_positions = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                h_positions.append((i, j))
    
    INF = float('inf')
    dist = [[INF] * W for _ in range(H)]
    q = deque()
    
    # Initialize distances for H cells and enqueue them
    for i, j in h_positions:
        dist[i][j] = 0
        q.append((i, j))
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while q:
        i, j = q.popleft()
        current_dist = dist[i][j]
        
        for di, dj in directions:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                # Skip walls
                if grid[ni][nj] == '#':
                    continue
                new_dist = current_dist + 1
                # Only proceed if the new distance is better and within D
                if new_dist < dist[ni][nj] and new_dist <= D:
                    dist[ni][nj] = new_dist
                    q.append((ni, nj))
    
    # Count the humidified cells
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '#' and dist[i][j] <= D:
                count += 1
    print(count)

if __name__ == "__main__":
    main()