import collections

def solve():
    n = int(input())
    if n == 2:
        u1, v1, l1 = map(int, input().split())
        length = l1
        print(2 * length)
        print(2 * length)
        return
        
    edges = []
    for _ in range(n - 1):
        u, v, l = map(int, input().split())
        edges.append(((u, v), l))
    
    results = []
    for k_val in range(1, n + 1):
        max_score = 0
        import itertools
        
        vertices = list(range(1, n + 1))
        best_score_for_k = 0
        
        vertex_sets = itertools.combinations(vertices, k_val)
        
        for s_tuple in vertex_sets:
            aoki_vertices = set(s_tuple)
            target_vertices = aoki_vertices.union({1})
            
            min_walk_length = 0
            
            # Find minimal subtree containing target_vertices
            subtree_edges = set()
            for v1 in target_vertices:
                for v2 in target_vertices:
                    if v1 == v2:
                        continue
                    path = []
                    q = collections.deque([(1, [], set())])
                    visited_paths = set()
                    found_path = False
                    while q:
                        current_vertex, current_path, visited_vertices = q.popleft()
                        if current_vertex == v1:
                            path_v1_to_v2 = []
                            
                            q2 = collections.deque([(v1, [], visited_vertices.copy())])
                            visited_paths_v1_v2 = set()
                            
                            while q2:
                                current_v_v2, current_path_v1_v2, visited_v_v2 = q2.popleft()
                                if current_v_v2 == v2:
                                    path_v1_to_v2 = current_path_v1_v2
                                    found_path = True
                                    break
                                visited_v_v2.add(current_v_v2)
                                for (u_e, v_e), length_e in edges:
                                    neighbor = -1
                                    if u_e == current_v_v2 and v_e not in visited_v_v2:
                                        neighbor = v_e
                                    elif v_e == current_v_v2 and u_e not in visited_v_v2:
                                        neighbor = u_e
                                    if neighbor != -1:
                                        if tuple(sorted((current_v_v2, neighbor))) not in visited_paths_v1_v2:
                                            visited_paths_v1_v2.add(tuple(sorted((current_v_v2, neighbor))))
                                            new_path_v1_v2 = current_path_v1_v2 + [((min(current_v_v2, neighbor), max(current_v_v2, neighbor)), length_e)]
                                            q2.append((neighbor, new_path_v1_v2, visited_v_v2.copy()))
                                if found_path:
                                    break
                            if found_path:
                                path.extend(path_v1_to_v2)
                                break
                                
                        visited_vertices.add(current_vertex)
                        for (u_e, v_e), length_e in edges:
                            neighbor = -1
                            if u_e == current_vertex and v_e not in visited_vertices:
                                neighbor = v_e
                            elif v_e == current_vertex and u_e not in visited_vertices:
                                neighbor = u_e
                            if neighbor != -1:
                                if tuple(sorted((current_vertex, neighbor))) not in visited_paths:
                                    visited_paths.add(tuple(sorted((current_vertex, neighbor))))
                                    new_path = current_path + [((min(current_vertex, neighbor), max(current_vertex, neighbor)), length_e)]
                                    q.append((neighbor, new_path, visited_vertices.copy()))
                        if found_path:
                            break

                    for edge_info in path:
                        subtree_edges.add(edge_info)
                        
            current_subtree_length = sum(length for _, length in subtree_edges)
            score = 2 * current_subtree_length
            best_score_for_k = max(best_score_for_k, score)
            
        results.append(best_score_for_k)
        
    for score in results:
        print(score)

if __name__ == '__main__':
    solve()