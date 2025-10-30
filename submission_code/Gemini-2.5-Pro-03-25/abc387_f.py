# YOUR CODE HERE
import sys

# Increase recursion depth for deep paths/large components
# Set a reasonably large limit like N + buffer. N <= 2025.
try:
    # Set recursion depth limit higher than default (usually 1000)
    # Needs to be large enough for the maximum path length in the graph's component structure.
    # N + ~500 seems safe enough for typical functional graph structures.
    sys.setrecursionlimit(2025 + 500) 
except OverflowError: 
    # Handle potential OS limits if setting recursion depth fails.
    # This might occur on systems with strict limits or specific configurations.
    # If it fails, the program will run with the system default limit, which might be insufficient.
    pass 

MOD = 998244353

def solve():
    N, M = map(int, sys.stdin.readline().split())
    
    # Handle edge case of N=0 although constraints say N >= 1.
    # If N=0, there's one sequence (the empty sequence) which vacuously satisfies the condition.
    if N == 0: 
        print(1) 
        return
        
    A = list(map(int, sys.stdin.readline().split()))
    # Convert to 0-based indexing
    A = [x - 1 for x in A] 

    # State tracking for graph traversal and component identification
    visited = [0] * N # 0: unvisited, 1: visiting (on current DFS path), 2: fully visited
    component_id = [-1] * N # Assigns each vertex to a component ID
    is_cycle_node = [False] * N # Flags vertices that are part of a cycle
    
    # Stores the representative vertex index for each component's cycle.
    # Using the minimum index vertex within the cycle as the representative.
    cycle_representatives = {} 
    component_count = 0

    # Find cycles and identify components using iterative DFS-like approach
    for i in range(N):
        if visited[i] == 0:
            path = [] # Track nodes on the current path starting from i
            curr = i
            
            # Phase 1: Trace path until a cycle is found or a previously visited node is encountered
            while visited[curr] == 0:
                visited[curr] = 1 # Mark as 'visiting'
                path.append(curr)
                # Ensure curr index is valid based on constraints
                curr = A[curr]

            # Phase 2: Process based on the state of the node `curr` where the path ended
            if visited[curr] == 1: # Found a cycle involving `curr` because it's still 'visiting'
                
                # Extract cycle vertices from the path list
                cycle_start_idx = -1
                try:
                    # Find the first occurrence of `curr` in the path
                    cycle_start_idx = path.index(curr) 
                except ValueError:
                    # This indicates `curr` was not in path, which contradicts `visited[curr]==1`.
                    # Should not happen in a valid functional graph under this traversal logic.
                    # If it did, it would suggest an issue with state management or graph structure assumptions.
                    pass # Assume standard functional graph behavior

                if cycle_start_idx != -1:
                    current_cycle_nodes = path[cycle_start_idx:]
                    
                    # Use the minimum index vertex in the cycle as its representative
                    representative = min(current_cycle_nodes)
                    cycle_representatives[component_count] = representative

                    # Assign component ID and mark cycle properties for nodes in the cycle
                    for node in current_cycle_nodes:
                        component_id[node] = component_count
                        is_cycle_node[node] = True
                        visited[node] = 2 # Mark as fully visited

                    # Assign component ID and mark properties for tree nodes leading into this cycle
                    for k in range(cycle_start_idx):
                        node = path[k]
                        component_id[node] = component_count
                        visited[node] = 2 # Mark as fully visited
                    
                    component_count += 1
                else:
                    # If `curr` wasn't found in path (error case), attempt recovery by marking path nodes as visited
                     for node in path:
                         if visited[node] == 1: visited[node] = 2 
                     pass


            elif visited[curr] == 2: # Path merged into an already fully visited component
                if component_id[curr] != -1: # Ensure the target node has a valid component ID
                    target_comp_id = component_id[curr]
                    
                    # Assign all nodes on the current path to this existing component
                    for node in path:
                        component_id[node] = target_comp_id
                        visited[node] = 2 # Mark as fully visited
                else:
                    # If `visited[curr]==2` but `component_id` is invalid, this suggests an issue.
                    # Mark path nodes as visited to prevent potential infinite loops/reprocessing.
                     for node in path:
                          if visited[node] == 1: visited[node] = 2 
                     pass
            
            # Clean up `visited` state for nodes marked 'visiting' but not processed as part of cycle/merge.
            # This ensures nodes that started the path but didn't lead to cycle detection themselves are marked done.
            for node in path:
                if visited[node] == 1: # Was marked 'visiting' but path terminated before cycle involving it
                     visited[node] = 2 # Mark as fully visited


    # --- Condensation Graph and Dynamic Programming Setup ---

    # Identify unique nodes for DP state (tree nodes + one representative per cycle)
    unique_node_indices = []
    # Stores whether a unique node represents a cycle. Keyed by node index.
    node_is_cycle = {} 
    
    for i in range(N):
        # Check if node 'i' was processed and assigned a component ID. Skip if not.
        if component_id[i] == -1: continue 

        if not is_cycle_node[i]:
            # Tree nodes are unique nodes in the condensation graph
            unique_node_indices.append(i)
            node_is_cycle[i] = False
        else:
            # For cycle nodes, only the designated representative is included as a unique node
            if component_id[i] not in cycle_representatives: continue # Safety check
            rep = cycle_representatives[component_id[i]]
            if i == rep:
                unique_node_indices.append(i)
                node_is_cycle[i] = True

    # Build predecessor list for condensation graph nodes based on original graph edges
    # `pred_map[u]` will store a list of node IDs `v` such that there's an edge `v -> u` in G'
    pred_map = {idx: [] for idx in unique_node_indices}
    
    for j in range(N): # Vertex `j` is the source of edge `(j, A[j])` in the original graph `G`
        if component_id[j] == -1: continue # Skip unprocessed nodes

        target_i = A[j]
        if component_id[target_i] == -1: continue # Skip if target is unprocessed

        # Map source vertex `j` to its corresponding node ID in the condensation graph `G'`
        if not is_cycle_node[j]:
            pred_node_id = j # Tree node maps to itself
        else:
            if component_id[j] not in cycle_representatives: continue # Safety check
            pred_node_id = cycle_representatives[component_id[j]] # Cycle node maps to its representative

        # Map target vertex `target_i` to its corresponding node ID in `G'`
        if not is_cycle_node[target_i]:
            target_node_id = target_i
        else:
            if component_id[target_i] not in cycle_representatives: continue # Safety check
            target_node_id = cycle_representatives[component_id[target_i]]

        # Determine if the edge `(j, A[j])` represents an edge internal to a cycle
        is_internal = is_cycle_node[j] and is_cycle_node[target_i] and component_id[j] == component_id[target_i]
        
        # If the edge is not internal to a cycle, it corresponds to an edge in `G'`
        if not is_internal:
             # The target node must be one of the unique nodes we track in DP state
             if target_node_id in pred_map:
                  # Add the predecessor node ID if it's not already listed for this target
                  if pred_node_id not in pred_map[target_node_id]:
                       pred_map[target_node_id].append(pred_node_id)

    # --- Dynamic Programming Calculation ---
    
    # Memoization tables for DP states
    dp_memo = {} # Stores results for `get_dp(u, k)`
    S_memo = {}  # Stores results for `get_S(u, k)`

    # `get_S(u, k)` computes Sum_{k'=1..k} dp[u][k'] mod MOD
    # Represents the number of ways to assign values to the subgraph rooted at `u` such that `x_u <= k`.
    def get_S(u, k):
        state = (u, k)
        if state in S_memo: return S_memo[state]
        if k == 0: return 0
        
        # Compute S[u][k] = S[u][k-1] + dp[u][k] using memoized values
        # Base case S[u][0] = 0 handled above.
        res = (get_S(u, k-1) + get_dp(u, k)) % MOD
        S_memo[state] = res
        return res

    # `get_dp(u, k)` computes dp[u][k] = Product_{v in Pred(u)} S[v][k] mod MOD
    # Represents the number of ways to assign values to the subgraph rooted at `u` such that `x_u = k`.
    def get_dp(u, k):
        state = (u, k)
        if state in dp_memo: return dp_memo[state]
        
        # Base case: If node `u` has no predecessors in G', the product is empty, yielding 1.
        prod = 1
        # Get predecessors for node `u` from the precomputed `pred_map`
        if u in pred_map:
             for v in pred_map[u]:
                 # Recursively compute S[v][k] needed for the product
                 prod = (prod * get_S(v, k)) % MOD
        
        dp_memo[state] = prod
        return prod

    # Final answer calculation:
    # The total number of ways is the product of ways for each independent component.
    # The number of ways for a component is given by S[cycle_rep][M], where cycle_rep is the root of the component DP structure.
    total_ways = 1
    processed_comp_roots = set() # Track processed components to avoid redundant calculations

    for comp_id in range(component_count):
        # Check if the component was fully processed and has a representative assigned
        if comp_id not in cycle_representatives: continue 

        cycle_rep_idx = cycle_representatives[comp_id]
        
        # Calculate ways for this component only if its root hasn't been processed yet
        if cycle_rep_idx not in processed_comp_roots:
            # Compute S(cycle_rep_idx, M) which triggers the recursive DP calculation for the component
            ways_for_comp = get_S(cycle_rep_idx, M)
            total_ways = (total_ways * ways_for_comp) % MOD
            processed_comp_roots.add(cycle_rep_idx)

    print(total_ways)

solve()