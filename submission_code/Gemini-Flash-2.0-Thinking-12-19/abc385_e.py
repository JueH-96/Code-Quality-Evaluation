import collections

def solve():
    n = int(input())
    if n == 1:
        print(0)
        return
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    min_deleted_vertices = n
    
    for i in range(1 << n):
        deleted_vertices_indices = []
        remaining_vertices = []
        original_indices_to_remaining_indices = {}
        remaining_index = 0
        
        for j in range(1, n + 1):
            if (i >> (j - 1)) & 1:
                deleted_vertices_indices.append(j)
            else:
                remaining_vertices.append(j)
                original_indices_to_remaining_indices[j] = remaining_index
                remaining_index += 1
                
        if not remaining_vertices:
            continue
            
        current_edges = []
        for u, v in edges:
            if u in remaining_vertices and v in remaining_vertices:
                u_index = original_indices_to_remaining_indices[u] + 1
                v_index = original_indices_to_remaining_indices[v] + 1
                current_edges.append((u_index, v_index))
                
        if not current_edges and len(remaining_vertices) > 1:
            continue
        if len(remaining_vertices) == 1:
            if len(deleted_vertices_indices) < min_deleted_vertices:
                min_deleted_vertices = len(deleted_vertices_indices)
            continue

        is_snowflake_tree = False
        
        if not current_edges and len(remaining_vertices) == 2:
            u1, u2 = remaining_vertices
            current_edges = [(original_indices_to_remaining_indices[u1]+1, original_indices_to_remaining_indices[u2]+1)]
            
        if not current_edges and len(remaining_vertices) == 3:
            u1, u2, u3 = remaining_vertices
            current_edges = [(original_indices_to_remaining_indices[u1]+1, original_indices_to_remaining_indices[u2]+1), (original_indices_to_remaining_indices[u2]+1, original_indices_to_remaining_indices[u3]+1)]


        if current_edges:
            num_remaining_vertices = len(remaining_vertices)
            
            degrees = collections.defaultdict(int)
            for u, v in current_edges:
                degrees[u] += 1
                degrees[v] += 1
                
            possible_centers = remaining_vertices
            
            for center_original_index in possible_centers:
                center_index = original_indices_to_remaining_indices[center_original_index] + 1
                neighbors = []
                for u, v in current_edges:
                    if u == center_index:
                        neighbors.append(v)
                    elif v == center_index:
                        neighbors.append(u)
                
                if not neighbors:
                    continue
                    
                x = len(neighbors)
                if x <= 0:
                    continue
                    
                y_val = -1
                valid_snowflake = True
                intermediate_vertices_indices = set()
                
                for neighbor_index in neighbors:
                    intermediate_vertices_indices.add(remaining_vertices[neighbor_index-1])
                    neighbor_degree = degrees[neighbor_index]
                    y = neighbor_degree - 1
                    if y <= 0:
                        valid_snowflake = False
                        break
                    if y_val == -1:
                        y_val = y
                    elif y_val != y:
                        valid_snowflake = False
                        break
                        
                if not valid_snowflake:
                    continue
                    
                if y_val <= 0:
                    continue
                    
                all_leaves = True
                for neighbor_index in neighbors:
                    intermediate_vertex_index = neighbor_index
                    for u_edge, v_edge in current_edges:
                        if u_edge == intermediate_vertex_index and v_edge != center_index:
                            if degrees[v_edge] != 1:
                                all_leaves = False
                                break
                        elif v_edge == intermediate_vertex_index and u_edge != center_index:
                            if degrees[u_edge] != 1:
                                all_leaves = False
                                break
                    if not all_leaves:
                        break
                        
                if all_leaves:
                    is_snowflake_tree = True
                    x_val = x
                    y_val_final = y_val
                    break
                    
        if is_snowflake_tree:
            deleted_count = len(deleted_vertices_indices)
            if deleted_count < min_deleted_vertices:
                min_deleted_vertices = deleted_count

    print(min_deleted_vertices)

if __name__ == '__main__':
    solve()