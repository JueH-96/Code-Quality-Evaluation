import sys
from collections import deque

def main():
    # Read H and W
    H_W = sys.stdin.readline().strip()
    while H_W == '':
        H_W = sys.stdin.readline().strip()
    H, W = map(int, H_W.split())
    
    # Read the grid
    grid = []
    for _ in range(H):
        line = sys.stdin.readline().strip()
        while line == '':
            line = sys.stdin.readline().strip()
        grid.append(list(line))
    
    # Define the sequence
    sequence = ['s', 'n', 'u', 'k', 'e']
    
    # Initialize visited array
    visited = [ [[False]*5 for _ in range(W)] for _ in range(H) ]
    
    # Initialize queue
    queue = deque()
    
    # Starting position
    if grid[0][0] == 's':
        queue.append( (0, 0, 0) )
        visited[0][0][0] = True
    
    # Define directions: up, down, left, right
    directions = [ (-1,0), (1,0), (0,-1), (0,1) ]
    
    # BFS
    while queue:
        i, j, k = queue.popleft()
        
        # Check if reached the target cell
        if i == H-1 and j == W-1:
            print("Yes")
            return
        
        # Explore all directions
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                next_k = (k + 1) % 5
                if grid[ni][nj] == sequence[next_k] and not visited[ni][nj][next_k]:
                    queue.append( (ni, nj, next_k) )
                    visited[ni][nj][next_k] = True
    
    # If target not reached
    print("No")

if __name__ == "__main__":
    main()