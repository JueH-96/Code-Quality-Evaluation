def main():
    import sys
    import heapq
    input = sys.stdin.readline

    # Read N: number of stages.
    n = int(input())
    
    # Construct graph for stages 1,2,...,N.
    # For each i (1 <= i <= N-1), we have two possible actions:
    #   Action 1: Move from stage i to stage i+1 with cost A_i.
    #   Action 2: Move from stage i to stage X_i with cost B_i.
    # We'll use a 1-indexed list of lists for the graph.
    graph = [[] for _ in range(n+1)]
    for stage in range(1, n):
        a, b, x = map(int, input().split())
        # Action 1: clear stage i and proceed to stage i+1.
        graph[stage].append((stage + 1, a))
        # Action 2: clear stage i and go to stage X_i.
        graph[stage].append((x, b))
    
    # Use Dijkstra's algorithm to compute the minimum total seconds to reach stage N.
    INF = 10**18
    dist = [INF] * (n+1)
    dist[1] = 0  # starting at stage 1 with 0 seconds elapsed.
    heap = [(0, 1)]
    
    while heap:
        current_time, stage = heapq.heappop(heap)
        
        if current_time != dist[stage]:
            continue
        
        if stage == n:
            # We've reached stage N.
            print(current_time)
            return
        
        for next_stage, time_cost in graph[stage]:
            next_time = current_time + time_cost
            if next_time < dist[next_stage]:
                dist[next_stage] = next_time
                heapq.heappush(heap, (next_time, next_stage))
    
    # Print the result.
    print(dist[n])
    
if __name__ == "__main__":
    main()