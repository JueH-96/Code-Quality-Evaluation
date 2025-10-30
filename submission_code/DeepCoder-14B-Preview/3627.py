import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        if n == 0:
            return 0
        m = len(moveTime[0])
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        heap = []
        heapq.heappush(heap, (0, 0, 0))
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while heap:
            current_time, i, j = heapq.heappop(heap)
            
            if i == n - 1 and j == m - 1:
                return current_time
            
            if current_time > dist[i][j]:
                continue
            
            for dx, dy in directions:
                x = i + dx
                y = j + dy
                if 0 <= x < n and 0 <= y < m:
                    new_time = max(current_time + 1, moveTime[x][y])
                    if new_time < dist[x][y]:
                        dist[x][y] = new_time
                        heapq.heappush(heap, (new_time, x, y))
        
        return dist[n-1][m-1]