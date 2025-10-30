from collections import deque, defaultdict

def bfs_shortest_path(graph, start, end, n):
    """Find the shortest path from start to end in terms of number of edges."""
    queue = deque([(start, 0)])
    distance = [-1] * (n + 1)
    distance[start] = 0
    
    while queue:
        node, dist = queue.popleft()
        for neighbor in graph[node]:
            if distance[neighbor] == -1:
                distance[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    
    return distance[end] if distance[end] != -1 else float('inf')

def max_flow(graph, source, sink, n):
    """Compute the maximum flow from source to sink using Edmonds-Karp."""
    # Build capacity matrix
    capacity = [[0] * (n + 1) for _ in range(n + 1)]
    for u in graph:
        for v in graph[u]:
            capacity[u][v] += 1
    
    flow = [[0] * (n + 1) for _ in range(n + 1)]
    total_flow = 0
    
    while True:
        # BFS to find an augmenting path
        parent = [-1] * (n + 1)
        visited = [False] * (n + 1)
        queue = deque([source])
        visited[source] = True
        
        while queue:
            u = queue.popleft()
            for v in range(1, n + 1):
                if not visited[v] and capacity[u][v] > flow[u][v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                    if v == sink:
                        break
        
        if not visited[sink]:
            break
        
        # Find the maximum flow through the path
        path_flow = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, capacity[u][v] - flow[u][v])
            v = u
        
        # Update the flow
        v = sink
        while v != source:
            u = parent[v]
            flow[u][v] += path_flow
            flow[v][u] -= path_flow
            v = u
        
        total_flow += path_flow
    
    return total_flow

def solve():
    n, m, k = map(int, input().split())
    graph = defaultdict(list)
    
    for _ in range(m):
        u, v = map(int, input().split())
        graph[u].append(v)
    
    # Find the shortest path length from 1 to n
    shortest_path_length = bfs_shortest_path(graph, 1, n, n)
    
    if shortest_path_length == float('inf'):
        print(0)
        return
    
    # Find the edge connectivity from 1 to n (max flow)
    edge_connectivity = max_flow(graph, 1, n, n)
    
    if edge_connectivity == 0:
        print(0)
        return
    
    # The answer is min(k // edge_connectivity, shortest_path_length)
    answer = min(k // edge_connectivity, shortest_path_length)
    
    print(answer)

solve()