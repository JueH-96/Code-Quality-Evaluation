from heapq import heappush, heappop

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        # Directions for moving to adjacent cells
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Priority queue: (current_time, x, y)
        heap = []
        heappush(heap, (0, 0, 0))
        # Visited array to keep track of the earliest time to reach each cell
        visited = [[float('inf')] * m for _ in range(n)]
        visited[0][0] = 0
        
        while heap:
            current_time, x, y = heappop(heap)
            if x == n - 1 and y == m - 1:
                return current_time
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m:
                    # The earliest time we can move to (nx, ny) is max(current_time + 1, moveTime[nx][ny])
                    new_time = max(current_time + 1, moveTime[nx][ny])
                    if new_time < visited[nx][ny]:
                        visited[nx][ny] = new_time
                        heappush(heap, (new_time, nx, ny))
        return -1  # Should not reach here as per constraints