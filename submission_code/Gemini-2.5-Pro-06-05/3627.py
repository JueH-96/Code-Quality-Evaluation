import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        
        # dist[r][c] stores the minimum time to arrive at cell (r, c).
        # Initialize with a value representing infinity.
        dist = [[float('inf')] * m for _ in range(n)]
        
        # The priority queue will store tuples of (time, row, col).
        # We start at room (0, 0) at time 0.
        pq = [(0, 0, 0)]
        dist[0][0] = 0
        
        # Directions for moving to adjacent rooms (up, down, left, right).
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while pq:
            time, r, c = heapq.heappop(pq)
            
            # If we've found a shorter path to (r, c) already, skip this one.
            if time > dist[r][c]:
                continue
            
            # If we've reached the destination, return the time.
            # Since Dijkstra's explores paths in increasing order of cost (time),
            # the first time we reach the destination will be via the shortest path.
            if r == n - 1 and c == m - 1:
                return time
            
            # Explore neighbors of the current room.
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                # Check if the neighbor is within the grid boundaries.
                if 0 <= nr < n and 0 <= nc < m:
                    # `time` is the arrival time at the current room (r, c).
                    # We must wait until at least `moveTime[nr][nc]` to start moving
                    # to the neighbor room (nr, nc).
                    start_move_time = max(time, moveTime[nr][nc])
                    
                    # The move itself takes 1 second.
                    new_time = start_move_time + 1
                    
                    # If this new path to the neighbor is shorter than any previously found path,
                    # update its minimum time and add it to the priority queue to explore later.
                    if new_time < dist[nr][nc]:
                        dist[nr][nc] = new_time
                        heapq.heappush(pq, (new_time, nr, nc))
                        
        return -1 # Should be unreachable given the problem constraints.