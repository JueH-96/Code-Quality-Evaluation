from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n, m = len(moveTime), len(moveTime[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def is_valid(x, y):
            return 0 <= x < n and 0 <= y < m
        
        pq = [(0, 0, 0)]  # (time, x, y)
        visited = set()
        
        while pq:
            time, x, y = heapq.heappop(pq)
            
            if (x, y) in visited:
                continue
            visited.add((x, y))
            
            if (x, y) == (n - 1, m - 1):
                return time
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny) and (nx, ny) not in visited:
                    next_time = max(time + 1, moveTime[nx][ny])
                    heapq.heappush(pq, (next_time, nx, ny))
        
        return -1  # Should never reach here if input is valid

# Example usage:
# sol = Solution()
# print(sol.minTimeToReach([[0,4],[4,4]]))  # Output: 6
# print(sol.minTimeToReach([[0,0,0],[0,0,0]]))  # Output: 3
# print(sol.minTimeToReach([[0,1],[1,2]]))  # Output: 3