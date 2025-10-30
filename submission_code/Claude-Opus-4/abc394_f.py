# YOUR CODE HERE
from collections import defaultdict, deque

def solve():
    n = int(input())
    
    # Build adjacency list
    adj = defaultdict(list)
    for _ in range(n - 1):
        a, b = map(int, input().split())
        adj[a].append(b)
        adj[b].append(a)
    
    max_alkane_size = -1
    
    # Try each vertex as a potential degree-4 vertex
    for start in range(1, n + 1):
        if len(adj[start]) < 4:
            continue
            
        # BFS to find maximum alkane with 'start' as a degree-4 vertex
        queue = deque([(start, frozenset([start]), frozenset())])
        visited_states = set()
        
        while queue:
            curr, vertices, edges = queue.popleft()
            
            state = (curr, vertices, edges)
            if state in visited_states:
                continue
            visited_states.add(state)
            
            # Check if current subgraph is a valid alkane
            degrees = defaultdict(int)
            for u, v in edges:
                degrees[u] += 1
                degrees[v] += 1
            
            is_valid_alkane = True
            has_degree_4 = False
            
            for v in vertices:
                deg = degrees[v]
                if deg == 4:
                    has_degree_4 = True
                elif deg != 1 and deg != 0:
                    is_valid_alkane = False
                    break
            
            if is_valid_alkane and has_degree_4:
                # Check if it's a tree (connected and |E| = |V| - 1)
                if len(edges) == len(vertices) - 1:
                    max_alkane_size = max(max_alkane_size, len(vertices))
            
            # Try to extend the alkane
            for v in vertices:
                if degrees[v] < 4:
                    for neighbor in adj[v]:
                        if neighbor not in vertices:
                            new_vertices = vertices | {neighbor}
                            new_edges = edges | {(min(v, neighbor), max(v, neighbor))}
                            
                            # Check if the new configuration could lead to a valid alkane
                            new_degrees = defaultdict(int)
                            for u, w in new_edges:
                                new_degrees[u] += 1
                                new_degrees[w] += 1
                            
                            valid_extension = True
                            for u in new_vertices:
                                if new_degrees[u] > 4:
                                    valid_extension = False
                                    break
                            
                            if valid_extension:
                                queue.append((neighbor, new_vertices, new_edges))
    
    print(max_alkane_size)

solve()