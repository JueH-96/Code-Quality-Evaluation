# YOUR CODE HERE
import sys
input = sys.stdin.read
from collections import deque

def main():
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    Y = int(data[2])
    
    A = []
    index = 3
    for i in range(H):
        A.append([int(data[index + j]) for j in range(W)])
        index += W
    
    # Directions for moving in the grid (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Function to perform BFS and mark all connected cells that will sink
    def bfs(sea_level):
        visited = [[False] * W for _ in range(H)]
        queue = deque()
        
        # Add all boundary cells to the queue if they are below or at sea level
        for i in range(H):
            for j in [0, W-1]:
                if A[i][j] <= sea_level and not visited[i][j]:
                    queue.append((i, j))
                    visited[i][j] = True
        for j in range(W):
            for i in [0, H-1]:
                if A[i][j] <= sea_level and not visited[i][j]:
                    queue.append((i, j))
                    visited[i][j] = True
        
        # Perform BFS
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and A[nx][ny] <= sea_level:
                    visited[nx][ny] = True
                    queue.append((nx, ny))
        
        # Count remaining land area
        remaining_area = 0
        for i in range(H):
            for j in range(W):
                if not visited[i][j]:
                    remaining_area += 1
        return remaining_area
    
    # Calculate the remaining area for each year
    results = []
    for year in range(1, Y + 1):
        sea_level = year
        remaining_area = bfs(sea_level)
        results.append(remaining_area)
    
    # Print the results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()