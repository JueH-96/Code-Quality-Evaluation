import heapq
import collections

def solve():
    N, M, K_budget = map(int, input().split())
    adj = collections.defaultdict(list)
    for _ in range(M):
        u, v = map(int, input().split())
        adj[u - 1].append(v - 1)

    # check(D_target) returns true if it's possible to make shortest path >= D_target
    # using at most K_budget edges with weight 1.
    def check(D_target):
        # dp[v][l] = min number of edges that must be chosen as weight 1
        #            to reach vertex v with a path of length l (l weight-1 edges)
        # v from 0 to N-1
        # l from 0 to N-1 (max path length in terms of edges is N-1)
        dp = [[float('inf')] * N for _ in range(N)]

        # Priority queue stores (cost, vertex, path_len_in_ones)
        # cost = number of edges chosen to be weight 1
        pq = [(0, 0, 0)]  # (cost_val, vertex_idx, path_len_val)
        dp[0][0] = 0

        while pq:
            cost, u, path_len = heapq.heappop(pq)

            if cost > dp[u][path_len]:
                continue
            
            # If u is destination N-1, and path_len >= D_target, we note this cost.
            # However, Dijkstra explores shortest paths based on 'cost'.
            # We are interested in min cost for path_len >= D_target.
            # This will be calculated after Dijkstra finishes.

            for v_neighbor in adj[u]:
                # Option 1: use edge (u, v_neighbor) as weight 0
                # Cost does not change, path_len does not change.
                if cost < dp[v_neighbor][path_len]:
                    dp[v_neighbor][path_len] = cost
                    heapq.heappush(pq, (cost, v_neighbor, path_len))

                # Option 2: use edge (u, v_neighbor) as weight 1
                # Cost increases by 1, path_len increases by 1.
                if path_len + 1 < N: # path_len can go up to N-1
                    if cost + 1 < dp[v_neighbor][path_len + 1]:
                        dp[v_neighbor][path_len + 1] = cost + 1
                        heapq.heappush(pq, (cost + 1, v_neighbor, path_len + 1))
        
        min_k_needed = float('inf')
        for l in range(D_target, N): # Path length must be at least D_target
            min_k_needed = min(min_k_needed, dp[N-1][l])
        
        return min_k_needed <= K_budget

    ans = 0
    low = 0
    high = N -1 # Max possible shortest path length

    # It is guaranteed N is reachable from 1.
    # If D_target = 0, min_k_needed will be 0 (path of all 0-cost edges).
    # This is always <= K_budget. So D_target=0 is always possible.
    
    while low <= high:
        mid = (low + high) // 2
        if mid == 0: # Base case $D=0$ always possible
            if check(mid): # check(0) should always be true
                 ans = max(ans, mid)
                 low = mid + 1
            else: # Should not happen for D=0
                 high = mid -1
            continue

        if check(mid):
            ans = max(ans, mid)
            low = mid + 1
        else:
            high = mid - 1
            
    print(ans)

solve()