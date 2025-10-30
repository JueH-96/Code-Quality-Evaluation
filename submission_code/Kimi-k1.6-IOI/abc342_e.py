import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    # Initialize trains grouped by their destination station B_i
    trains_by_b = [[] for _ in range(N + 1)]  # 1-based indexing
    for _ in range(M):
        l = int(data[idx])
        idx += 1
        d = int(data[idx])
        idx += 1
        k = int(data[idx])
        idx += 1
        c = int(data[idx])
        idx += 1
        a = int(data[idx])
        idx += 1
        b = int(data[idx])
        idx += 1
        trains_by_b[b].append((a, l, d, k, c))
    
    # Compute dp[N] as the maximum arrival time of any train ending at N
    INF = -1 << 60  # A very small number to represent -infinity
    dp = [INF] * (N + 1)
    max_arrival = INF
    for train in trains_by_b[N]:
        a, l, d, k, c = train
        last_departure = l + (k - 1) * d
        arrival_time = last_departure + c
        if arrival_time > max_arrival:
            max_arrival = arrival_time
    dp[N] = max_arrival
    
    # Priority queue (min-heap storing negative values to simulate max-heap)
    heap = []
    if dp[N] != INF:
        heapq.heappush(heap, (-dp[N], N))
    
    # Process the heap
    while heap:
        current_neg_dp, u = heapq.heappop(heap)
        current_dp = -current_neg_dp
        
        # Skip if this entry is outdated
        if current_dp != dp[u]:
            continue
        
        # Iterate through all trains ending at u (B_i = u)
        for train in trains_by_b[u]:
            a, l, d, k, c = train
            X = current_dp - c
            if X < l:
                continue
            
            # Calculate the maximum valid m
            m = (X - l) // d
            m = min(m, k - 1)
            t_candidate = l + m * d
            
            # Update dp[a] if t_candidate is better
            if t_candidate > dp[a]:
                dp[a] = t_candidate
                heapq.heappush(heap, (-dp[a], a))
    
    # Prepare the output
    results = []
    for s in range(1, N):
        if dp[s] == INF:
            results.append("Unreachable")
        else:
            results.append(str(dp[s]))
    
    print('
'.join(results))

if __name__ == "__main__":
    main()