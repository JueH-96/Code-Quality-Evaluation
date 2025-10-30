from collections import deque

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n, m = len(moveTime), len(moveTime[0])
        queue = deque([(0, 0, 0)])  # (row, col, time)
        visited = set()
        
        while queue:
            row, col, time = queue.popleft()
            
            if row == n - 1 and col == m - 1:
                return time
            
            if (row, col) in visited:
                continue
            
            visited.add((row, col))
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < n and 0 <= new_col < m:
                    new_time = max(time, moveTime[new_row][new_col]) + 1
                    queue.append((new_row, new_col, new_time))
        
        return -1