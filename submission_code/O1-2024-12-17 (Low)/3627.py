class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        import heapq
        
        n = len(moveTime)
        m = len(moveTime[0])
        
        # Directions for moving up, down, left, right
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        
        # dist will hold the minimum time to reach each cell
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0  # we start at (0,0) at time 0
        
        # Min-heap (time, row, col)
        pq = [(0, 0, 0)]
        
        while pq:
            current_time, r, c = heapq.heappop(pq)
            
            # If we've popped a cell with a time worse than our best known, skip it
            if current_time > dist[r][c]:
                continue
            
            # If we've reached the target room, return the time
            if r == n - 1 and c == m - 1:
                return current_time
            
            # Explore all 4 directions
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < m:
                    # We can only start to move into (nr, nc) if time >= moveTime[nr][nc]
                    # Then we spend 1 second moving.
                    next_time = max(current_time, moveTime[nr][nc]) + 1
                    if next_time < dist[nr][nc]:
                        dist[nr][nc] = next_time
                        heapq.heappush(pq, (next_time, nr, nc))
        
        # If we exit the loop without returning, theoretically we didn't reach the end
        return dist[n - 1][m - 1]