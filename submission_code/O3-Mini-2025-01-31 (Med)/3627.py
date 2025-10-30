from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n, m = len(moveTime), len(moveTime[0])
        # dist will hold the minimum time when we arrive at each cell
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        # Priority queue: (arrival_time, x, y)
        pq = [(0, 0, 0)]
        
        # Dijkstra's algorithm on grid
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while pq:
            current_time, x, y = heapq.heappop(pq)
            # If we have already found a better route, skip
            if current_time > dist[x][y]:
                continue
            # If we reached the bottom-right room, return the time
            if x == n - 1 and y == m - 1:
                return current_time
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m:
                    # You can only start moving into neighbor at time >= moveTime[nx][ny]
                    # so wait if necessary
                    start_move_time = max(current_time, moveTime[nx][ny])
                    arrival_time = start_move_time + 1  # Moving takes exactly one second
                    if arrival_time < dist[nx][ny]:
                        dist[nx][ny] = arrival_time
                        heapq.heappush(pq, (arrival_time, nx, ny))
                        
        # Return -1 if no path exists; though according to constraints, there is always a path.
        return -1

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1
    moveTime1 = [[0,4],[4,4]]
    print(sol.minTimeToReach(moveTime1))  # Output: 6
    
    # Example 2
    moveTime2 = [[0,0,0],[0,0,0]]
    print(sol.minTimeToReach(moveTime2))  # Output: 3

    # Example 3
    moveTime3 = [[0,1],[1,2]]
    print(sol.minTimeToReach(moveTime3))  # Output: 3