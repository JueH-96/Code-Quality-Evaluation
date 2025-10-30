class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        import heapq
        
        n, m = len(moveTime), len(moveTime[0])
        
        # Dijkstra's algorithm
        pq = [(0, 0, 0)]  # (time, row, col)
        visited = set()
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while pq:
            time, row, col = heapq.heappop(pq)
            
            if (row, col) in visited:
                continue
            visited.add((row, col))
            
            if row == n - 1 and col == m - 1:
                return time
            
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < n and 0 <= nc < m and (nr, nc) not in visited:
                    next_time = max(time, moveTime[nr][nc]) + 1
                    heapq.heappush(pq, (next_time, nr, nc))
        
        return -1  # Should not reach here given the constraints