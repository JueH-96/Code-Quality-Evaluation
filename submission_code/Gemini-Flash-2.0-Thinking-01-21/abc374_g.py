import collections

def solve():
    n = int(input())
    s_list = [input() for _ in range(n)]
    if not s_list:
        print(0)
        return
    
    edges = set()
    vertices = set()
    for s in s_list:
        u, v = s[0], s[1]
        edges.add((u, v))
        vertices.add(u)
        vertices.add(v)
        
    if not edges:
        print(0)
        return
        
    vertex_list = list(vertices)
    vertex_to_index = {v: i for i, v in enumerate(vertex_list)}
    index_to_vertex = {i: v for i, v in enumerate(vertex_list)}
    
    adjacency_list = collections.defaultdict(list)
    for u, v in edges:
        adjacency_list[u].append(v)
        
    components = []
    visited_vertices = set()
    
    for vertex in vertices:
        if vertex not in visited_vertices:
            component_vertices = set()
            component_edges = set()
            queue = collections.deque([vertex])
            visited_vertices.add(vertex)
            component_vertices.add(vertex)
            
            while queue:
                current_vertex = queue.popleft()
                for neighbor in adjacency_list:
                    if neighbor == current_vertex:
                        for v_neighbor in adjacency_list[neighbor]:
                            component_edges.add((neighbor, v_neighbor))
                            if v_neighbor not in visited_vertices and v_neighbor in vertices:
                                visited_vertices.add(v_neighbor)
                                component_vertices.add(v_neighbor)
                                queue.append(v_neighbor)
                    elif neighbor in component_vertices:
                        for v_neighbor in adjacency_list[neighbor]:
                            if v_neighbor in component_vertices:
                                component_edges.add((neighbor, v_neighbor))
                                
            components.append({'vertices': component_vertices, 'edges': component_edges})
            
    total_ng_strings = 0
    for component_info in components:
        component_edges = component_info['edges']
        if not component_edges:
            continue
            
        in_degree = collections.defaultdict(int)
        out_degree = collections.defaultdict(int)
        component_vertices = component_info['vertices']
        
        component_vertex_set = set(component_vertices)
        component_edge_list = list(component_edges)
        
        for u, v in component_edge_list:
            if u in component_vertex_set and v in component_vertex_set:
                out_degree[u] += 1
                in_degree[v] += 1
                
        source_count = 0
        sink_count = 0
        
        for vertex in component_vertices:
            diff = out_degree[vertex] - in_degree[vertex]
            if diff > 0:
                source_count += diff
            elif diff < 0:
                sink_count -= diff
                
        paths_needed = 0
        if source_count > 0:
            paths_needed = max(1, source_count)
        elif sink_count > 0:
            paths_needed = max(1, sink_count)
        elif component_edges:
            paths_needed = 1
        else:
            paths_needed = 0
            
        total_ng_strings += paths_needed
        
    print(total_ng_strings)

if __name__ == '__main__':
    solve()