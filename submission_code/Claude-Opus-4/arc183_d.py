from collections import defaultdict, deque

def find_distance(graph, u, v, n):
    if u == v:
        return 0
    
    dist = [-1] * (n + 1)
    dist[u] = 0
    queue = deque([u])
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if dist[neighbor] == -1:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
                if neighbor == v:
                    return dist[neighbor]
    
    return -1

def can_remove_pair(graph, u, v, n):
    # Check if removing u and v maintains perfect matching
    # The key insight is that after removing two vertices,
    # the remaining graph should still have even components
    # where each component has a perfect matching
    
    # Create a copy of the graph without u and v
    temp_graph = defaultdict(list)
    for node in range(1, n + 1):
        if node != u and node != v:
            for neighbor in graph[node]:
                if neighbor != u and neighbor != v:
                    temp_graph[node].append(neighbor)
    
    # Check if each connected component has even size
    visited = [False] * (n + 1)
    visited[u] = visited[v] = True
    
    for start in range(1, n + 1):
        if not visited[start]:
            # BFS to find component size
            component_size = 0
            queue = deque([start])
            visited[start] = True
            
            while queue:
                node = queue.popleft()
                component_size += 1
                for neighbor in temp_graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            if component_size % 2 != 0:
                return False
    
    return True

def solve():
    n = int(input())
    graph = defaultdict(list)
    
    for _ in range(n - 1):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    
    removed = [False] * (n + 1)
    result = []
    
    for _ in range(n // 2):
        # Find all current leaves
        leaves = []
        for node in range(1, n + 1):
            if not removed[node]:
                degree = sum(1 for neighbor in graph[node] if not removed[neighbor])
                if degree == 1:
                    leaves.append(node)
        
        # Find the best pair of leaves to remove
        best_pair = None
        best_distance = -1
        
        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                u, v = leaves[i], leaves[j]
                
                # Check if this pair can be removed
                if can_remove_pair(graph, u, v, n):
                    dist = find_distance(graph, u, v, n)
                    if dist > best_distance:
                        best_distance = dist
                        best_pair = (u, v)
        
        # Remove the best pair
        if best_pair:
            u, v = best_pair
            removed[u] = removed[v] = True
            result.append((u, v))
            
            # Update the graph by marking removed vertices
            for node in range(1, n + 1):
                if not removed[node]:
                    graph[node] = [neighbor for neighbor in graph[node] if not removed[neighbor]]
    
    # Print the result
    for u, v in result:
        print(u, v)

solve()