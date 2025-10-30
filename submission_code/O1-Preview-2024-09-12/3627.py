class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        import heapq

        n = len(moveTime)
        m = len(moveTime[0])
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        heap = [(0, 0, 0)]  # (arrival time, x, y)

        while heap:
            t_arrive, x, y = heapq.heappop(heap)

            if t_arrive > dist[x][y]:
                continue  # Already found a better path

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m:
                    # Earliest time we can start moving to the next room
                    t_start_moving = max(t_arrive, moveTime[nx][ny])
                    # Arrival time at the next room
                    t_next_arrival = t_start_moving + 1
                    if t_next_arrival < dist[nx][ny]:
                        dist[nx][ny] = t_next_arrival
                        heapq.heappush(heap, (t_next_arrival, nx, ny))

        return dist[n - 1][m - 1]