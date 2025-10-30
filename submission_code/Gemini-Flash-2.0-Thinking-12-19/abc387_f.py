import sys
input = sys.stdin.readline
MOD = 998244353

N, M = map(int, input().split())
A = list(map(int, input().split()))

# Build reverse adjacency list for the full graph
rev_adj_full = [[] for _ in range(N + 1)]
for i in range(1, N + 1):
    rev_adj_full[A[i-1]].append(i)

# 1. Find cycles and identify cycle nodes
visited_trace = [0] * (N + 1) # 0=unvisited, 1=visiting, 2=visited/processed
is_cycle = [False] * (N + 1)
distinct_cycles_repr = set() # Store one node from each distinct cycle

def find_cycles_recursive(u, path):
    visited_trace[u] = 1
    path.append(u)
    v = A[u-1]
    
    # Ensure A[u-1] is within bounds, though constraints guarantee this
    if not (1 <= v <= N):
         # This case should not be reached based on constraints
         pass

    if visited_trace[v] == 1: # Found cycle
        cycle_start_idx = path.index(v)
        for node in path[cycle_start_idx:]:
            is_cycle[node] = True
        distinct_cycles_repr.add(path[cycle_start_idx]) # Use the first node in cycle from path as representative
    elif visited_trace[v] == 0:
        find_cycles_recursive(v, path)
    visited_trace[u] = 2
    path.pop()

for i in range(1, N + 1):
    if visited_trace[i] == 0:
        find_cycles_recursive(i, [])

Total_ways = 1

# Keep track of nodes already assigned to a component
visited_component_nodes = set()

# Process each component
# Each component is uniquely identified by its cycle.
# Iterate through the representatives of distinct cycles found.
for cycle_repr in list(distinct_cycles_repr): # Iterate over a copy because we modify visited_component_nodes
    if cycle_repr in visited_component_nodes:
        continue

    # Find all nodes in this component using BFS on the reversed graph
    nodes_in_comp = set()
    q_comp = [cycle_repr]
    visited_comp_bfs = {cycle_repr}
    head = 0
    while head < len(q_comp):
        u = q_comp[head]
        head += 1
        nodes_in_comp.add(u)
        # Mark node as visited for component processing
        visited_component_nodes.add(u)

        # Add nodes pointing to u in the original graph (reversed edges)
        for v in rev_adj_full[u]:
            if v not in visited_comp_bfs:
                 visited_comp_bfs.add(v)
                 q_comp.append(v)

    # nodes_in_comp now contains all nodes in this component
    U_C = [u for u in nodes_in_comp if not is_cycle[u]]

    ways_for_this_component = 0

    if not U_C:
        # Component is just a cycle (or multiple nodes in a cycle).
        # Ways = Sum_{y=1..M} 1 = M.
        ways_for_this_component = M % MOD
    else:
        # Component has non-cycle nodes U_C.
        # Build reversed graph on U_C (vertices U_C, edge v <- u if A[u-1]=v, u,v in U_C)
        # and compute in-degrees in this reversed graph on U_C.
        # in_degree_Uc[v] = number of u in U_C such that A[u-1] = v
        rev_adj_Uc = {u: [] for u in U_C} # Children of u in reversed U_C graph
        in_degree_Uc = {u: 0 for u in U_C} # In-degree of u in reversed U_C graph
        U_C_set = set(U_C) # For O(1) lookup

        for u in U_C:
            v = A[u-1]
            if v in U_C_set: # v is also in U_C
                rev_adj_Uc[v].append(u) # v <- u in reversed U_C graph
                in_degree_Uc[v] += 1

        # Topological sort on reversed graph on U_C
        q_topo = []
        current_in_degree_Uc = {}
        for u in U_C:
            current_in_degree_Uc[u] = in_degree_Uc[u]
            if current_in_degree_Uc[u] == 0: # Leaves in reversed U_C graph
                q_topo.append(u)

        topo_order_Uc = []
        head_topo = 0
        while head_topo < len(q_topo):
            u = q_topo[head_topo]
            head_topo += 1
            topo_order_Uc.append(u)

            # Children of u in reversed U_C graph are w such that A[w-1] = u and w in U_C.
            # These are stored in rev_adj_Uc[u].
            for w in rev_adj_Uc[u]:
                current_in_degree_Uc[w] -= 1
                if current_in_degree_Uc[w] == 0:
                    q_topo.append(w)
        
        # Should check if len(topo_order_Uc) == len(U_C). If not, there's a cycle in U_C, which shouldn't happen.

        # Compute S[u][v] for u in U_C and v = 1..M
        # S_values_Uc[u][v_minus_1] stores S[u][v] (number of ways to assign values <= v)
        S_values_Uc = {}

        for u in topo_order_Uc: # Process leaves first
            S_values_Uc[u] = [0] * M
            children_w = rev_adj_Uc[u] # Children in the reversed U_C graph

            if not children_w: # u is a leaf in reversed U_C graph (in-degree 0 in reversed U_C graph)
                # S[u][v] = v for v=1..M
                for v_minus_1 in range(M):
                    S_values_Uc[u][v_minus_1] = v_minus_1 + 1
            else:
                dp_u = [0] * M # dp_u[v_minus_1] stores dp[u][v = v_minus_1 + 1]
                for v_minus_1 in range(M): # v = v_minus_1 + 1
                    v = v_minus_1 + 1
                    prod_S = 1
                    for w in children_w:
                        # S[w][v] = S_values_Uc[w][v-1] (using 0-based index for v-1)
                        prod_S = (prod_S * S_values_Uc[w][v_minus_1]) % MOD
                    dp_u[v_minus_1] = prod_S

                # Compute S[u][v] from dp[u][v] using prefix sums
                S_values_Uc[u][0] = dp_u[0]
                for v_minus_1 in range(1, M):
                    S_values_Uc[u][v_minus_1] = (S_values_Uc[u][v_minus_1 - 1] + dp_u[v_minus_1]) % MOD

        # Roots of the forward graph on U_C are the leaves of the reversed graph on U_C.
        # These are the nodes with original in_degree_Uc == 0.
        roots_Uc_forward_graph = [u for u in U_C if in_degree_Uc[u] == 0]

        # Calculate ways for component C by summing over cycle value y = 1..M
        # Total ways for non-cycle nodes U_C, given cycle value y, is
        # Product of S[u][y] for roots u of the forward graph on U_C.
        ways_for_this_component = 0
        for y in range(1, M + 1):
            ways_fixed_y_Uc = 1
            for u in roots_Uc_forward_graph:
                 # S[u][y] is S_values_Uc[u][y-1]
                 ways_fixed_y_Uc = (ways_fixed_y_Uc * S_values_Uc[u][y-1]) % MOD

            ways_for_this_component = (ways_for_this_component + ways_fixed_y_Uc) % MOD

    Total_ways = (Total_ways * ways_for_this_component) % MOD

print(Total_ways)