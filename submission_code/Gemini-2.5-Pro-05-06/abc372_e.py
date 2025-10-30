import sys

# Use fast I/O
input = sys.stdin.readline
print_ = sys.stdout.write

K_MAX = 10  # Max K elements to store for each component

# Iterative find with path compression
def find(i_idx, parent_arr):
    path_to_root = []
    curr = i_idx
    while parent_arr[curr] != curr:
        path_to_root.append(curr)
        curr = parent_arr[curr]
    
    # Path compression: make all nodes on path point directly to root
    root = curr
    for node_on_path in path_to_root:
        parent_arr[node_on_path] = root
    return root

def main():
    N, Q = map(int, input().split())

    # DSU data structures, 1-indexed
    parent = list(range(N + 1))  # parent[i] = i initially
    # top_k_elements[i] stores up to K_MAX largest vertices in component rooted at i
    # Each component initially has one vertex i, so its top_k list is just [i]
    top_k_elements = [[i] for i in range(N + 1)]  # top_k_elements[0] is dummy
    
    # For union-by-size heuristic
    component_sizes = [1] * (N + 1)  # component_sizes[i] = 1 initially; component_sizes[0] dummy

    results_buffer = []  # To store answers for Type 2 queries

    for _ in range(Q):
        query_parts = list(map(int, input().split()))
        query_type = query_parts[0]

        if query_type == 1:
            # Add an edge between u and v
            u, v_node = query_parts[1], query_parts[2] # Renamed v to v_node to avoid conflict
            
            root_u = find(u, parent)
            root_v = find(v_node, parent)

            if root_u != root_v:
                # Union operation. Use union-by-size:
                # Attach smaller tree to root of larger tree.
                if component_sizes[root_u] < component_sizes[root_v]:
                    # Ensure root_u is the root of the larger (or equal size) component
                    root_u, root_v = root_v, root_u 
                
                parent[root_v] = root_u
                component_sizes[root_u] += component_sizes[root_v]
                # component_sizes[root_v] becomes outdated/irrelevant.

                # Merge top_k_elements lists.
                list_surviving_root = top_k_elements[root_u]
                list_absorbed_root = top_k_elements[root_v]
                
                merged_list_for_new_root = []
                p1, p2 = 0, 0  # Pointers for list_surviving_root and list_absorbed_root
                
                # Sentinel value for exhausted lists; vertex IDs are >= 1.
                # Using a value guaranteed to be smaller than any valid vertex ID.
                sentinel = -1 

                while len(merged_list_for_new_root) < K_MAX:
                    val_from_list1 = sentinel 
                    if p1 < len(list_surviving_root):
                        val_from_list1 = list_surviving_root[p1]
                    
                    val_from_list2 = sentinel 
                    if p2 < len(list_absorbed_root):
                        val_from_list2 = list_absorbed_root[p2]

                    if val_from_list1 == sentinel and val_from_list2 == sentinel:
                        break # Both lists exhausted

                    if val_from_list1 >= val_from_list2:
                        merged_list_for_new_root.append(val_from_list1)
                        p1 += 1
                    else: # val_from_list2 > val_from_list1
                        merged_list_for_new_root.append(val_from_list2)
                        p2 += 1
                
                top_k_elements[root_u] = merged_list_for_new_root
                top_k_elements[root_v] = [] # Clear absorbed list 


        else: # query_type == 2
            v_node, k_val = query_parts[1], query_parts[2] # Renamed v to v_node
            root_of_v = find(v_node, parent)
            candidate_elements = top_k_elements[root_of_v]
            
            if k_val > len(candidate_elements):
                results_buffer.append("-1")
            else:
                results_buffer.append(str(candidate_elements[k_val - 1]))
                
    if results_buffer: # Only print if there are results for Type 2 queries
        print_("
".join(results_buffer) + "
")

if __name__ == '__main__':
    main()