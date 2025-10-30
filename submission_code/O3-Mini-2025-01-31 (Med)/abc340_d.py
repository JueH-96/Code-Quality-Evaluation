def main():
    import sys
    import heapq
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # Arrays to store the parameters for stages 1 through n-1 (1-indexed)
    A = [0] * (n+1)
    B = [0] * (n+1)
    X = [0] * (n+1)
    
    for i in range(1, n):
        A[i] = int(next(it))
        B[i] = int(next(it))
        X[i] = int(next(it))
    
    # Use Dijkstra's algorithm on a graph where stages 1..n are nodes.
    # From stage i (1 <= i <= n-1) there are two moves:
    #   1. i -> i+1 with cost A[i]
    #   2. i -> X[i] with cost B[i]
    INF = 10**18  # sufficiently large
    dist = [INF] * (n+1)
    dist[1] = 0
    # Priority queue stores (current_time, stage)
    pq = [(0, 1)]
    
    while pq:
        d, stage = heapq.heappop(pq)
        if d != dist[stage]:
            continue
        if stage == n:
            print(d)
            return
        
        # We only have moves if stage < n
        if stage < n:
            # Action 1: Clear stage i for A[i] seconds to unlock stage (i+1)
            nd = d + A[stage]
            next_stage = stage + 1
            if nd < dist[next_stage]:
                dist[next_stage] = nd
                heapq.heappush(pq, (nd, next_stage))
            # Action 2: Clear stage i for B[i] seconds to unlock stage X[i]
            nd = d + B[stage]
            next_stage = X[stage]
            if nd < dist[next_stage]:
                dist[next_stage] = nd
                heapq.heappush(pq, (nd, next_stage))
                
    # Since the problem guarantees that stage N can be reached, we should have returned already.
    print(dist[n])

if __name__ == '__main__':
    main()