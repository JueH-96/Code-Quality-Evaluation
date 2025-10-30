import collections

def solve():
    n = int(input())
    product_names = [input() for _ in range(n)]
    if not product_names:
        print(0)
        return
    
    edges = []
    vertices = set()
    for name in product_names:
        u, v = name[0], name[1]
        edges.append((u, v))
        vertices.add(u)
        vertices.add(v)
    
    vertex_list = sorted(list(vertices))
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
            component_edges = []
            queue = collections.deque([vertex])
            visited_vertices.add(vertex)
            component_vertices.add(vertex)
            while queue:
                u = queue.popleft()
                for v in adjacency_list[u]:
                    component_edges.append((u, v))
                    if v not in visited_vertices and v in vertices:
                        visited_vertices.add(v)
                        component_vertices.add(v)
                        queue.append(v)
                for v_u, v_v in edges:
                    if v_v == u and v_u in component_vertices and v_u not in visited_vertices:
                        visited_vertices.add(v_u)
                        component_vertices.add(v_u)
                        queue.append(v_u)
                        component_edges.append((v_u, v_v))

            components.append((component_vertices, component_edges))
            
    total_paths = 0
    for component_vertices, component_edges in components:
        if not component_edges:
            continue
        
        in_degree = collections.defaultdict(int)
        out_degree = collections.defaultdict(int)
        component_vertex_set = set(component_vertices)
        component_edge_list = []
        for u, v in edges:
            if u in component_vertex_set and v in component_vertex_set:
                component_edge_list.append((u, v))
                out_degree[u] += 1
                in_degree[v] += 1
                
        p_value = 0
        for vertex in component_vertices:
            diff = out_degree[vertex] - in_degree[vertex]
            if diff > 0:
                p_value += diff
                
        if p_value > 0:
            total_paths += p_value
        else:
            total_paths += 1
            
    print(total_paths)

if __name__ == '__main__':
    solve()