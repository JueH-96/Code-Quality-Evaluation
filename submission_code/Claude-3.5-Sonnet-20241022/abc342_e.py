from collections import defaultdict
from heapq import heappush, heappop

def solve():
    N, M = map(int, input().split())
    
    # Build graph of trains
    trains = defaultdict(list)
    for _ in range(M):
        l, d, k, c, a, b = map(int, input().split())
        # Store (departure_time, arrival_time, destination) for each source station
        for i in range(k):
            t = l + i * d
            trains[a].append((t, t + c, b))
    
    def get_max_arrival(start):
        # dp[station] = max arrival time at station
        dp = defaultdict(lambda: float('-inf'))
        # pq stores (-arrival_time, station)
        pq = []
        
        # Initialize with all trains from start station
        for dep_time, arr_time, dest in trains[start]:
            heappush(pq, (-arr_time, dest))
            dp[dest] = max(dp[dest], arr_time)
            
        while pq:
            neg_curr_time, curr = heappop(pq)
            curr_time = -neg_curr_time
            
            if curr_time < dp[curr]:
                continue
                
            for dep_time, arr_time, dest in trains[curr]:
                if dep_time >= curr_time:
                    if arr_time > dp[dest]:
                        dp[dest] = arr_time
                        heappush(pq, (-arr_time, dest))
        
        return dp[N]
    
    # Calculate f(s) for each starting station
    for s in range(1, N):
        result = get_max_arrival(s)
        if result == float('-inf'):
            print("Unreachable")
        else:
            print(result)

solve()