import sys

# It's good practice to set recursion limit for deep DFS.
# Max N, M, Q can be up to 4*10^5. Add some buffer.
sys.setrecursionlimit(4 * 10**5 + 500) 

def solve():
    N, M, Q = map(int, sys.stdin.readline().split())
    
    people_orig = []
    for i in range(M):
        s, t = map(int, sys.stdin.readline().split())
        # Adjust to 0-indexed towns
        people_orig.append({'id': i, 's': s - 1, 't': t - 1})

    queries_orig = []
    for i in range(Q):
        l, r = map(int, sys.stdin.readline().split())
        queries_orig.append({'id': i, 'l': l - 1, 'r': r - 1})

    ans_array = [""] * Q

    # Variables for DFS cycle detection, defined outside loop to reuse memory
    # (though Python lists might reallocate anyway)
    adj = []
    visited_dfs = []
    recursion_stack = []

    # DFS utility for cycle detection
    # u is a component's dense ID
    def has_cycle_util(u_comp_id, current_adj_list):
        nonlocal visited_dfs, recursion_stack # Ensure we're modifying the outer scope lists
        visited_dfs[u_comp_id] = True
        recursion_stack[u_comp_id] = True

        for v_comp_id in current_adj_list[u_comp_id]:
            if not visited_dfs[v_comp_id]:
                if has_cycle_util(v_comp_id, current_adj_list):
                    return True
            elif recursion_stack[v_comp_id]: # If v_comp_id is in recursion stack, cycle found
                return True
        
        recursion_stack[u_comp_id] = False
        return False

    # DSU parent array
    parent = list(range(N))
    def find_set(v_town_idx):
        if v_town_idx == parent[v_town_idx]:
            return v_town_idx
        parent[v_town_idx] = find_set(parent[v_town_idx])
        return parent[v_town_idx]

    def unite_sets(a_town_idx, b_town_idx):
        a_root = find_set(a_town_idx)
        b_root = find_set(b_town_idx)
        if a_root != b_root:
            # Basic union: make a_root parent of b_root. Could use union by size/rank.
            parent[b_root] = a_root
            return True
        return False

    for query_info in queries_orig:
        query_idx = query_info['id']
        L_person_idx, R_person_idx = query_info['l'], query_info['r']
        
        # Reset DSU for each query by re-initializing parent array
        for i in range(N):
            parent[i] = i
        
        current_people_in_query = people_orig[L_person_idx : R_person_idx + 1]

        for person in current_people_in_query:
            unite_sets(person['s'], person['t'])

        possible = True
        # Local conflict check
        for person in current_people_in_query:
            s_town, t_town = person['s'], person['t']
            root_S = find_set(s_town)

            if s_town < t_town:
                for p_intermediate in range(s_town + 1, t_town):
                    if find_set(p_intermediate) == root_S:
                        possible = False
                        break
            else: # s_town > t_town
                for p_intermediate in range(t_town + 1, s_town):
                    if find_set(p_intermediate) == root_S:
                        possible = False
                        break
            if not possible:
                break
        
        if not possible:
            ans_array[query_idx] = "No"
            continue

        # Global conflict check (cycles in component graph)
        # Map component roots (0 to N-1 values) to dense IDs (0 to num_comp-1)
        comp_root_to_dense_id = {}
        num_distinct_components = 0
        # Ensure all N town roots are considered for mapping, even if isolated.
        # This is implicitly handled if we only map roots that appear in constraints.
        # We only need to map roots that are find(s) or find(p) for some person.
        # For safety, map all find(i) for i=0..N-1.
        
        # Efficiently find all unique roots and map them
        unique_roots = []
        # Not N, but only towns involved for people L..R matter. However, any town can be intermediate.
        # So all N towns' components must be considered.
        for i in range(N): 
            root_i = find_set(i)
            if root_i not in comp_root_to_dense_id:
                comp_root_to_dense_id[root_i] = num_distinct_components
                unique_roots.append(root_i) # Keep track of unique roots if needed
                num_distinct_components += 1
        
        # Resize adj lists and DFS visited arrays if num_distinct_components changed
        # Python lists grow dynamically, but for fixed-size arrays (like in other languages)
        # one would ensure they are large enough. Here, recreate them based on num_distinct_components.
        
        # Using sets for adjacency lists to auto-handle duplicate edges
        # This can be slow due to hashing and set operations.
        temp_adj_sets = [set() for _ in range(num_distinct_components)]

        for person in current_people_in_query:
            s_town, t_town = person['s'], person['t']
            
            # Get dense ID for component of S
            comp_S_dense_id = comp_root_to_dense_id[find_set(s_town)]

            if s_town < t_town:
                for p_intermediate in range(s_town + 1, t_town):
                    root_P = find_set(p_intermediate)
                    if find_set(s_town) != root_P: # Edge only if different components
                        comp_P_dense_id = comp_root_to_dense_id[root_P]
                        # Edge: comp_S -> comp_P (W_comp_P >= W_comp_S + 1)
                        temp_adj_sets[comp_S_dense_id].add(comp_P_dense_id) 
            else: # s_town > t_town
                for p_intermediate in range(t_town + 1, s_town):
                    root_P = find_set(p_intermediate)
                    if find_set(s_town) != root_P: # Edge only if different components
                        comp_P_dense_id = comp_root_to_dense_id[root_P]
                        # Edge: comp_P -> comp_S (W_comp_S >= W_comp_P + 1)
                        temp_adj_sets[comp_P_dense_id].add(comp_S_dense_id)
        
        # Prepare adj list for DFS (list of lists is faster for iteration)
        # Resize adj if necessary based on num_distinct_components
        if len(adj) < num_distinct_components: # Ensure adj is large enough
            adj = [[] for _ in range(num_distinct_components)]
        for i in range(num_distinct_components):
            adj[i] = list(temp_adj_sets[i])

        # Resize visited arrays for DFS
        if len(visited_dfs) < num_distinct_components: # Ensure visited_dfs is large enough
            visited_dfs = [False] * num_distinct_components
            recursion_stack = [False] * num_distinct_components
        else: # Reset for current query
            for i in range(num_distinct_components):
                visited_dfs[i] = False
                recursion_stack[i] = False
        
        has_cycle_in_graph = False
        for i in range(num_distinct_components):
            if not visited_dfs[i]:
                if has_cycle_util(i, adj):
                    has_cycle_in_graph = True
                    break
        
        if has_cycle_in_graph:
            ans_array[query_idx] = "No"
        else:
            ans_array[query_idx] = "Yes"

    sys.stdout.write("
".join(ans_array) + "
")

solve()