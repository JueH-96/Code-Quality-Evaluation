import sys
from collections import deque

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    snuke = 'snuke'
    
    # Check if the starting cell is 's'
    if grid[0][0] != 's':
        print("No")
        return
    
    # Initialize visited array and queue
    visited = [[[False for _ in range(5)] for _ in range(W)] for _ in range(H)]
    queue = deque()
    queue.append((0, 0, 0))
    visited[0][0][0] = True
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        i, j, phase = queue.popleft()
        
        # Check if current cell is the destination
        if i == H - 1 and j == W - 1:
            print("Yes")
            return
        
        # Calculate next phase and character
        next_phase = (phase + 1) % 5
        next_char = snuke[next_phase]
        
        # Explore all four directions
        for di, dj in directions:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == next_char and not visited[ni][nj][next_phase]:
                    visited[ni][nj][next_phase] = True
                    queue.append((ni, nj, next_phase))
    
    # If no path found
    print("No")

if __name__ == "__main__":
    main()