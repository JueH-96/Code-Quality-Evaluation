import heapq

def main():
    INF = float('inf')
    N, M = map(int, input().split())
    edges = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v, w = map(int, input().split())
        edges[u].append((v, w))
    
    # Initialize the DP table
    dp = [[INF] * (N + 1) for _ in range(1 << N)]
    
    # Priority queue: (current_cost, current_node, current_mask)
    heap = []
    
    for u in range(1, N+1):
        mask = 1 << (u - 1)
        dp[mask][u] = 0
        heapq.heappush(heap, (0, u, mask))
    
    while heap:
        current_cost, u, mask = heapq.heappop(heap)
        
        # If we've found a cheaper way already, skip
        if current_cost > dp[mask][u]:
            continue
        
        # Process each outgoing edge
        for v, w in edges[u]:
            new_mask = mask | (1 << (v - 1))
            new_cost = current_cost + w
            
            if new_cost < dp[new_mask][v]:
                dp[new_mask][v] = new_cost
                heapq.heappush(heap, (new_cost, v, new_mask))
    
    full_mask = (1 << N) - 1
    min_total = INF
    for u in range(1, N+1):
        if dp[full_mask][u] < min_total:
            min_total = dp[full_mask][u]
    
    if min_total == INF:
        print("No")
    else:
        print(min_total)

if __name__ == "__main__":
    main()