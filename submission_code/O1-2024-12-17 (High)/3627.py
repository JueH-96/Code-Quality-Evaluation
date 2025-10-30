class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        import heapq
        
        n = len(moveTime)
        m = len(moveTime[0])
        
        # Distance array to keep track of the earliest time
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        
        # Min-heap for Dijkstra-like approach; stores (time, row, col)
        pq = [(0, 0, 0)]
        
        # Directions for adjacent cells (up, down, left, right)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        while pq:
            curr_time, x, y = heapq.heappop(pq)
            
            # If this is not the best time for (x, y), skip
            if curr_time > dist[x][y]:
                continue
            
            # If we've reached the destination, return the time
            if x == n - 1 and y == m - 1:
                return curr_time
            
            # Explore neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m:
                    # Must wait until the current time is at least moveTime[nx][ny]
                    # Then spend 1 second to move
                    next_time = max(curr_time, moveTime[nx][ny]) + 1
                    if next_time < dist[nx][ny]:
                        dist[nx][ny] = next_time
                        heapq.heappush(pq, (next_time, nx, ny))
        
        # If somehow we didn't reach the destination in the loop, return the distance
        return dist[n - 1][m - 1]