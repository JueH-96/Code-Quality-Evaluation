import sys
from collections import deque

def main():
    H, W, Y = map(int, sys.stdin.readline().split())
    A = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]
    
    # Initialize the sea level and the area
    sea_level = 0
    area = H * W
    
    # Create a list to store the results for each year
    results = []
    
    # Create a priority queue to process the sections in order of their elevation
    # We will use a list of lists, where each list contains the coordinates of sections with the same elevation
    elevation_map = {}
    for i in range(H):
        for j in range(W):
            if A[i][j] not in elevation_map:
                elevation_map[A[i][j]] = []
            elevation_map[A[i][j]].append((i, j))
    
    # Sort the elevations in ascending order
    sorted_elevations = sorted(elevation_map.keys())
    
    # Initialize the current sea level and the area
    current_sea_level = 0
    current_area = H * W
    
    # Initialize the queue for BFS
    queue = deque()
    
    # Initialize the visited matrix
    visited = [[False for _ in range(W)] for _ in range(H)]
    
    # For each year, process the sections that are at or below the new sea level
    for year in range(1, Y+1):
        new_sea_level = year
        # Process all elevations up to the new sea level
        while sorted_elevations and sorted_elevations[0] <= new_sea_level:
            elevation = sorted_elevations.pop(0)
            for (i, j) in elevation_map[elevation]:
                if not visited[i][j]:
                    # Mark as visited and add to the queue
                    visited[i][j] = True
                    queue.append((i, j))
                    # Decrease the area
                    current_area -= 1
            # Perform BFS to sink adjacent sections
            while queue:
                x, y = queue.popleft()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if not visited[nx][ny] and A[nx][ny] <= new_sea_level:
                            visited[nx][ny] = True
                            queue.append((nx, ny))
                            current_area -= 1
        # Append the current area to the results
        results.append(current_area)
    
    # Print the results
    for res in results:
        print(res)

if __name__ == "__main__":
    main()