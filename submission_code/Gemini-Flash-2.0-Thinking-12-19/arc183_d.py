import collections

def solve():
    n = int(input())
    edges_input = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges_input.append((u, v))
    
    edges = list(edges_input)
    vertices = set()
    for u, v in edges:
        vertices.add(u)
        vertices.add(v)
    
    current_vertices = set(vertices)
    current_edges = list(edges)
    output_pairs = []
    
    for _ in range(n // 2):
        adjacency_list = collections.defaultdict(list)
        for u, v in current_edges:
            adjacency_list[u].append(v)
            adjacency_list[v].append(u)
            
        leaves = []
        for v in current_vertices:
            if len(adjacency_list[v]) <= 1:
                leaves.append(v)
                
        if not leaves:
            break
            
        max_distance = -1
        best_pair = None
        
        import itertools
        leaf_pairs = list(itertools.combinations(leaves, 2))
        
        for u, v in leaf_pairs:
            if u not in current_vertices or v not in current_vertices:
                continue
                
            queue = collections.deque([(u, 0)])
            visited = {u}
            distances = {u: 0}
            path_found = False
            distance = -1
            
            while queue:
                current_vertex, current_distance = queue.popleft()
                if current_vertex == v:
                    distance = current_distance
                    path_found = True
                    break
                for neighbor in adjacency_list[current_vertex]:
                    if neighbor not in visited and neighbor in current_vertices:
                        visited.add(neighbor)
                        distances[neighbor] = current_distance + 1
                        queue.append((neighbor, current_distance + 1))
                        
            if path_found:
                if distance > max_distance:
                    max_distance = distance
                    best_pair = (u, v)
                    
        if best_pair:
            u_remove, v_remove = best_pair
            output_pairs.append(best_pair)
            
            next_vertices = set()
            for v in current_vertices:
                if v != u_remove and v != v_remove:
                    next_vertices.add(v)
            current_vertices = next_vertices
            
            next_edges = []
            for edge in current_edges:
                u_edge, v_edge = edge
                if u_edge != u_remove and u_edge != v_remove and v_edge != u_remove and v_edge != v_remove:
                    next_edges.append(edge)
            current_edges = next_edges
        else:
            break

    for pair in output_pairs:
        print(f"{pair[0]} {pair[1]}")

if __name__ == '__main__':
    solve()