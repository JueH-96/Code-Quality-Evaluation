import collections

def solve():
    n = int(input())
    if n == 2:
        input()
        print(1)
        return
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    adjacency_list = collections.defaultdict(list)
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
        
    def get_degree(current_adj, vertex):
        return len(current_adj[vertex])
        
    def get_leaves(current_adj, vertices):
        leaves = []
        for vertex in vertices:
            if get_degree(current_adj, vertex) <= 1:
                leaves.append(vertex)
        return leaves

    def solve_instance(current_edges, current_vertices):
        adj = collections.defaultdict(list)
        for u, v in current_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        vertices_set = set(current_vertices)
        operation_count = 0
        while 1 in vertices_set:
            if get_degree(adj, 1) <= 1:
                operation_count += 1
                vertices_set.remove(1)
                del adj[1]
                for v_list in adj.values():
                    if 1 in v_list:
                        v_list.remove(1)
                break
                
            leaves = get_leaves(adj, vertices_set)
            valid_leaves = [leaf for leaf in leaves if leaf != 1]
            if not valid_leaves:
                # If only leaves are vertex 1 or no leaves other than 1, and vertex 1 is not yet deleted, 
                # then vertex 1 must be a leaf or isolated. 
                # If degree of 1 is > 1, then something is wrong. But we checked degree of 1 at the beginning. 
                # It must be that vertex 1 is a leaf. 
                operation_count += 1
                vertices_set.remove(1)
                del adj[1]
                for v_list in adj.values():
                    if 1 in v_list:
                        v_list.remove(1)
                break

            leaf_to_delete = -1
            max_original_distance = -1
            
            distances_from_1_original = {}
            queue = collections.deque([(1, 0)])
            visited_original = {1}
            distances_from_1_original[1] = 0
            
            original_adjacency_list = collections.defaultdict(list)
            for u, v in edges:
                original_adjacency_list[u].append(v)
                original_adjacency_list[v].append(u)
                
            while queue:
                u_vertex, dist = queue.popleft()
                for neighbor in original_adjacency_list[u_vertex]:
                    if neighbor not in visited_original:
                        visited_original.add(neighbor)
                        distances_from_1_original[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
                        
            best_leaf = -1
            max_dist = -1
            
            for leaf in valid_leaves:
                dist = distances_from_1_original.get(leaf, -1)
                if dist > max_dist:
                    max_dist = dist
                    best_leaf = leaf
                elif dist == max_dist:
                    if best_leaf == -1 or leaf < best_leaf:
                        best_leaf = leaf
                        
            leaf_to_delete = best_leaf

            if leaf_to_delete == -1:
                # Fallback strategy, just pick any leaf != 1
                if valid_leaves:
                    leaf_to_delete = valid_leaves[0]
                else:
                    break # Should not happen if vertex 1 is not yet a leaf and not yet deleted. 

            operation_count += 1
            vertex_to_remove = leaf_to_delete
            vertices_set.remove(vertex_to_remove)
            del adj[vertex_to_remove]
            for v_list in adj.values():
                if vertex_to_remove in v_list:
                    v_list.remove(vertex_to_remove)

        return operation_count

    initial_vertices = list(range(1, n + 1))
    min_ops = solve_instance(edges, initial_vertices)
    print(min_ops)

if __name__ == '__main__':
    solve()