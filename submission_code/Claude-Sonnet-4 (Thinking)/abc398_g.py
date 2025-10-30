from collections import defaultdict, deque

def solve():
    n, m = map(int, input().split())
    
    graph = defaultdict(list)
    for _ in range(m):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
    
    visited = [False] * (n + 1)
    component_partitions = []
    
    for i in range(1, n + 1):
        if not visited[i]:
            # Find connected component starting from vertex i
            component = []
            queue = deque([i])
            visited[i] = True
            
            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            if len(component) == 1:
                # Isolated vertex: can be placed in either partition
                component_partitions.append((1, 0))
            else:
                # Find bipartite partitioning using BFS
                color = [-1] * (n + 1)
                color[component[0]] = 0
                queue = deque([component[0]])
                
                partition_0 = 0
                partition_1 = 0
                
                while queue:
                    node = queue.popleft()
                    if color[node] == 0:
                        partition_0 += 1
                    else:
                        partition_1 += 1
                    
                    for neighbor in graph[node]:
                        if color[neighbor] == -1:
                            color[neighbor] = 1 - color[node]
                            queue.append(neighbor)
                
                component_partitions.append((partition_0, partition_1))
    
    # Find the partition of vertices that maximizes the number of edges
    # We want to split vertices into two groups of sizes as close as possible to n/2
    target = n // 2
    
    # Use DP to find all possible sums
    dp = [False] * (n + 1)
    dp[0] = True
    
    for a, b in component_partitions:
        new_dp = dp[:]
        for i in range(n + 1):
            if dp[i]:
                if i + a <= n:
                    new_dp[i + a] = True
                if i + b <= n:
                    new_dp[i + b] = True
        dp = new_dp
    
    # Find the achievable sum closest to target
    best_left_size = -1
    best_dist = float('inf')
    for i in range(n + 1):
        if dp[i]:
            dist = abs(i - target)
            if dist < best_dist:
                best_dist = dist
                best_left_size = i
    
    left_size = best_left_size
    right_size = n - left_size
    max_edges = left_size * right_size
    moves = max_edges - m
    
    if moves % 2 == 1:
        print("Aoki")
    else:
        print("Takahashi")

solve()