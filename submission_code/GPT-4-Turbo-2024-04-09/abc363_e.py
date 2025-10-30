import sys
import collections

input = sys.stdin.read
def main():
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    Y = int(data[2])
    
    A = []
    index = 3
    for _ in range(H):
        A.append([int(data[index + j]) for j in range(W)])
        index += W
    
    # Directions for adjacency (left, right, up, down)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # To keep track of which cells are sunk
    sunk = [[False] * W for _ in range(H)]
    
    # Queue for BFS
    queue = collections.deque()
    
    # Initialize the queue with all border cells that are at or below sea level 0
    for i in range(H):
        for j in range(W):
            if (i == 0 or i == H-1 or j == 0 or j == W-1) and A[i][j] <= 0:
                queue.append((i, j))
                sunk[i][j] = True
    
    # Perform initial BFS to sink all connected cells at sea level 0
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and not sunk[nx][ny] and A[nx][ny] <= 0:
                sunk[nx][ny] = True
                queue.append((nx, ny))
    
    # Count initially non-sunk cells
    initial_count = sum(not sunk[i][j] for i in range(H) for j in range(W))
    
    # Results for each year
    results = []
    
    # Process each year
    for year in range(1, Y + 1):
        # Increase sea level, sink cells accordingly
        new_sinks = []
        for i in range(H):
            for j in range(W):
                if not sunk[i][j] and A[i][j] <= year:
                    # Check if it should sink due to adjacency to already sunk cells
                    should_sink = False
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < H and 0 <= nj < W and sunk[ni][nj]:
                            should_sink = True
                            break
                    if should_sink:
                        new_sinks.append((i, j))
        
        # Sink the new cells
        for i, j in new_sinks:
            if not sunk[i][j]:
                queue.append((i, j))
                sunk[i][j] = True
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and not sunk[nx][ny] and A[nx][ny] <= year:
                    sunk[nx][ny] = True
                    queue.append((nx, ny))
        
        # Count non-sunk cells
        remaining_count = sum(not sunk[i][j] for i in range(H) for j in range(W))
        results.append(remaining_count)
    
    # Output results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()