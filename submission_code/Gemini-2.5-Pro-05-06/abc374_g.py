import collections

def solve():
    N = int(input())
    edges_input = []
    for _ in range(N):
        edges_input.append(input())

    if N == 0:
        print(0)
        return

    adj = collections.defaultdict(list)
    rev_adj = collections.defaultdict(list) # For reversed graph traversals
    nodes = set() # All unique letters involved

    for s_edge in edges_input:
        u, v = s_edge[0], s_edge[1]
        adj[u].append(v)
        rev_adj[v].append(u) # u <- v, so v is a predecessor of u
        nodes.add(u)
        nodes.add(v)

    # Find Weakly Connected Components (WCCs)
    visited_wcc_global = set()
    wccs_node_sets = [] # List of sets, each set contains nodes of a WCC

    for node_initial in nodes: # Iterate over all nodes that appear in edges
        if node_initial not in visited_wcc_global:
            current_wcc_nodes = set()
            q = collections.deque()
            
            q.append(node_initial)
            visited_wcc_global.add(node_initial)
            current_wcc_nodes.add(node_initial)
            
            while q:
                u = q.popleft()
                # Explore using forward edges
                for v_neighbor in adj[u]:
                    if v_neighbor not in visited_wcc_global: # Check global set
                        visited_wcc_global.add(v_neighbor)
                        current_wcc_nodes.add(v_neighbor)
                        q.append(v_neighbor)
                # Explore using backward edges (as if graph is undirected)
                for v_neighbor in rev_adj[u]: # v_neighbor -> u
                    if v_neighbor not in visited_wcc_global:
                        visited_wcc_global.add(v_neighbor)
                        current_wcc_nodes.add(v_neighbor)
                        q.append(v_neighbor)
            
            wcc_has_edges = False
            for u_wcc in current_wcc_nodes:
                for v_wcc_neighbor in adj[u_wcc]:
                    if v_wcc_neighbor in current_wcc_nodes:
                        wcc_has_edges = True
                        break
                if wcc_has_edges:
                    break
            
            if wcc_has_edges:
                 wccs_node_sets.append(current_wcc_nodes)
    
    total_paths = 0

    for wcc_nodes in wccs_node_sets:
        if not wcc_nodes: # Should not happen due to wcc_has_edges check
            continue

        # Build subgraph for WCC: adj, rev_adj, degrees
        wcc_adj = collections.defaultdict(list)
        wcc_rev_adj = collections.defaultdict(list) # For SCC check on WCC
        wcc_deg_in = collections.defaultdict(int)
        wcc_deg_out = collections.defaultdict(int)

        for u in wcc_nodes:
            for v_neighbor in adj[u]:
                if v_neighbor in wcc_nodes: # Edge (u, v_neighbor) is within WCC
                    wcc_adj[u].append(v_neighbor)
                    wcc_deg_out[u] += 1
                    wcc_deg_in[v_neighbor] += 1
            for u_predecessor in rev_adj[u]: # u_predecessor -> u
                 if u_predecessor in wcc_nodes:
                    wcc_rev_adj[u].append(u_predecessor) # Used for reversed graph traversal

        # 1. Check if WCC is strongly connected
        start_node_scc_check = next(iter(wcc_nodes)) # Get an arbitrary node
        
        # DFS/BFS from start_node_scc_check in wcc_adj
        q_scc_dfs = collections.deque([start_node_scc_check])
        visited_scc_dfs = {start_node_scc_check}
        while q_scc_dfs:
            curr = q_scc_dfs.popleft()
            for neighbor in wcc_adj[curr]:
                if neighbor not in visited_scc_dfs:
                    visited_scc_dfs.add(neighbor)
                    q_scc_dfs.append(neighbor)
        
        is_scc = (len(visited_scc_dfs) == len(wcc_nodes))
        
        if is_scc:
            # DFS/BFS from start_node_scc_check in wcc_rev_adj (reversed graph)
            q_scc_dfs_rev = collections.deque([start_node_scc_check])
            visited_scc_dfs_rev = {start_node_scc_check}
            while q_scc_dfs_rev:
                curr = q_scc_dfs_rev.popleft()
                for neighbor in wcc_rev_adj[curr]: # These are v such that v -> curr
                    if neighbor not in visited_scc_dfs_rev:
                        visited_scc_dfs_rev.add(neighbor)
                        q_scc_dfs_rev.append(neighbor)
            if len(visited_scc_dfs_rev) != len(wcc_nodes):
                is_scc = False
        
        if is_scc:
            total_paths += 1
            continue

        # 2. Not SCC. Check if DAG.
        # Iterative DFS for cycle detection for current WCC
        is_dag = True
        # path_dfs_states: 0=unvisited, 1=visiting (on stack), 2=visited (finished)
        # For this check, we only need to distinguish on_current_path vs visited_overall
        # visited_overall_dag_check: nodes whose DFS tree is fully explored (globally for this WCC)
        # on_current_path_stack: nodes in the current DFS path/stack
        
        visited_dag_dfs_overall = set() # Nodes whose processing is complete for DAG check
        
        for node_dag_check_start in wcc_nodes:
            if node_dag_check_start not in visited_dag_dfs_overall:
                # DFS from node_dag_check_start
                # stack_for_dfs stores (node, iterator_for_neighbors)
                dfs_iter_stack = [(node_dag_check_start, iter(wcc_adj[node_dag_check_start]))]
                on_current_path_dfs = {node_dag_check_start} 

                while dfs_iter_stack:
                    curr_node, children_iter = dfs_iter_stack[-1]
                    
                    try:
                        child = next(children_iter)
                        if child in on_current_path_dfs: # Cycle detected
                            is_dag = False 
                            break 
                        # If child not visited_dag_dfs_overall, it means it's not from a completed DFS tree
                        # We only care if it's on current path for cycle.
                        # If already in visited_dag_dfs_overall, it's a cross/forward edge to an already processed part, not a cycle FOR THIS PATH.
                        if child not in visited_dag_dfs_overall: 
                            # visited_dag_dfs_overall.add(child) # Mistake: add when popped / fully explored
                            on_current_path_dfs.add(child)
                            dfs_iter_stack.append((child, iter(wcc_adj[child])))
                    except StopIteration: # No more children for curr_node
                        # Finished with curr_node, remove from current path and add to overall visited
                        on_current_path_dfs.remove(curr_node) 
                        visited_dag_dfs_overall.add(curr_node) 
                        dfs_iter_stack.pop()
                if not is_dag: # If cycle found, break from checking other start nodes
                    break
        
        if is_dag:
            num_sources_in_wcc = 0
            for node_in_wcc in wcc_nodes:
                if wcc_deg_in[node_in_wcc] == 0:
                    num_sources_in_wcc += 1
            # If a WCC is a DAG and has edges, it must have at least one source.
            total_paths += num_sources_in_wcc
            continue

        # 3. Not SCC, not DAG
        sum_net_outflow = 0
        for node_in_wcc in wcc_nodes:
            sum_net_outflow += max(0, wcc_deg_out[node_in_wcc] - wcc_deg_in[node_in_wcc])
        
        total_paths += max(1, sum_net_outflow)

    print(total_paths)

solve()