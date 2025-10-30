def solve():
    import sys
    input = sys.stdin.read
    from collections import defaultdict, deque
    
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    edges = []
    index = 2
    for _ in range(M):
        U = int(data[index]) - 1
        V = int(data[index + 1]) - 1
        edges.append((U, V))
        index += 2
    
    # Graph adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # To keep track of visited nodes and their colors
    visited = [-1] * N
    
    def bfs(start):
        queue = deque([start])
        visited[start] = 0
        count = [0, 0]
        count[0] += 1
        edge_count = 0
        
        while queue:
            node = queue.popleft()
            current_color = visited[node]
            
            for neighbor in graph[node]:
                edge_count += 1
                if visited[neighbor] == -1:
                    visited[neighbor] = 1 - current_color
                    count[visited[neighbor]] += 1
                    queue.append(neighbor)
                elif visited[neighbor] == current_color:
                    # This should not happen as the graph is initially bipartite
                    raise ValueError("Graph is not bipartite")
        
        # Each edge is counted twice in undirected graph
        edge_count //= 2
        return count[0], count[1], edge_count
    
    total_possible_edges = 0
    
    for node in range(N):
        if visited[node] == -1:
            c1, c2, existing_edges = bfs(node)
            total_possible_edges += c1 * c2 - existing_edges
    
    if total_possible_edges % 2 == 1:
        print("Aoki")
    else:
        print("Takahashi")