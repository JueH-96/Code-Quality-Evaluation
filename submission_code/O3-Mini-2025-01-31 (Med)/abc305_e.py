def main():
    import sys
    from collections import deque
    input = sys.stdin.readline

    # Read first line containing N, M, K
    N, M, K = map(int, input().split())
    
    # Build the graph as an adjacency list
    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
        
    # dp[v] will store the maximum remaining guard power that can cover vertex v.
    # A vertex is guarded if dp[v] is at least 0 (which means there is a guard 
    # whose distance to v is within his stamina).
    dp = [-1] * (N + 1)
    dq = deque()
    
    # For each guard, initialize its starting vertex with its stamina.
    # (Each guard only appears once since p_i are distinct.)
    for _ in range(K):
        p, h = map(int, input().split())
        if h > dp[p]:
            dp[p] = h
            dq.append(p)
    
    # Multi-source BFS: propagate the guard influence throughout the graph.
    while dq:
        v = dq.popleft()
        current_power = dp[v]
        if current_power <= 0:
            continue
        # Try to update all neighbors: passing along one less power.
        for nb in graph[v]:
            next_power = current_power - 1
            if next_power > dp[nb]:
                dp[nb] = next_power
                dq.append(nb)
    
    # A vertex is guarded if dp[v] is at least 0 (i.e., some guard reached it)
    guarded = [v for v in range(1, N + 1) if dp[v] >= 0]
    guarded.sort()
    
    # Print the output in the requested format.
    sys.stdout.write(str(len(guarded)) + "
")
    if guarded:
        sys.stdout.write(" ".join(map(str, guarded)) + "
")

if __name__ == '__main__':
    main()