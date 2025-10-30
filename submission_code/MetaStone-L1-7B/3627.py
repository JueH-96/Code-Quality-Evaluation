import heapq

def minTimeToReach(matrix: List[List[int]]) -> int:
    n = len(matrix)
    if n == 0:
        return 0
    m = len(matrix[0])
    if m == 0:
        return 0
    
    # Initialize distance matrix with infinity
    dist = [[float('inf')] * m for _ in range(n)]
    dist[0][0] = 0
    
    # Priority queue: (distance, row, col)
    heap = []
    heapq.heappush(heap, (0, 0, 0))
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while heap:
        current_dist, i, j = heapq.heappop(heap)
        if current_dist > dist[i][j]:
            continue  # Skip if a better path was already found
        
        for dx, dy in directions:
            x = i + dx
            y = j + dy
            if 0 <= x < n and 0 <= y < m:
                # Compute new_time using the target cell's moveTime
                new_time = max(current_dist, matrix[x][y]) + 1
                if new_time < dist[x][y]:
                    dist[x][y] = new_time
                    heapq.heappush(heap, (new_time, x, y))
    
    return dist[n-1][m-1]