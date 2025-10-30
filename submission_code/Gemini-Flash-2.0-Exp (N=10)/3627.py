import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        
        pq = [(0, 0, 0)]  # (time, row, col)
        
        while pq:
            time, row, col = heapq.heappop(pq)
            
            if time > dist[row][col]:
                continue
            
            if row == n - 1 and col == m - 1:
                return time
            
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < n and 0 <= new_col < m:
                    wait_time = max(0, moveTime[new_row][new_col] - time)
                    new_time = time + wait_time + 1
                    
                    if new_time < dist[new_row][new_col]:
                        dist[new_row][new_col] = new_time
                        heapq.heappush(pq, (new_time, new_row, new_col))
        
        return -1 # Should not reach here based on problem constraints