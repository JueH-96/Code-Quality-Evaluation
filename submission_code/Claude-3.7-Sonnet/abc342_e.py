def solve():
    N, M = map(int, input().split())
    
    # Build a reverse graph: from arrival stations to departure stations
    graph = [[] for _ in range(N+1)]
    for _ in range(M):
        l, d, k, c, a, b = map(int, input().split())
        graph[b].append((a, l, d, k, c))  # (departure_station, start_time, interval, count, travel_time)
    
    # Initialize latest arrival times
    f = [-float('inf')] * (N + 1)
    f[N] = float('inf')  # We're already at the destination
    
    # Fixed-point iteration to compute f(S) for all stations
    for _ in range(N - 1):  # At most N-1 iterations
        changed = False
        for t in range(1, N + 1):
            if f[t] == -float('inf'):
                continue  # We can't reach the destination from station t
            
            for s, l, d, k, c in graph[t]:
                # Compute the latest valid departure time for this train
                if f[t] < l + c:  # Even the first train arrival exceeds f[t]
                    continue
                    
                # Find the latest train that arrives at station t on time
                I = min(k - 1, (f[t] - l - c) // d)
                
                if I >= 0:  # We have a valid departure
                    if f[s] < f[t]:
                        f[s] = f[t]
                        changed = True
        
        if not changed:
            break
    
    # Output the results
    for i in range(1, N):
        if f[i] == -float('inf'):
            print("Unreachable")
        else:
            print(f[i])

solve()