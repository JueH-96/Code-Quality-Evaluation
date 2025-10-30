import sys
from collections import deque

# Set recursion depth limit to handle potentially deep paths/trees
# Maximum path length is N.
# For cycle detection DFS, stack depth can be N.
# For BFS/in-degree, stack/queue depth is manageable.
# The DP part iterates based on topological sort, max depth N.
# Recursion is not explicitly used in the final O(NM) logic.
# However, underlying structure finding might use recursion.
# Set it high enough just in case, though iterative might be safer.
# With N=2025, 3000 should be enough for simple path following or DFS states.
sys.setrecursionlimit(3000)

MOD = 998244353

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    # Adjust A to be 0-indexed
    A = [x - 1 for x in A]

    # 1. Build reversed graph
    rev_adj = [[] for _ in range(N)]
    for i in range(N):
        rev_adj[A[i]].append(i)

    # 2. Find cycles and tree nodes using original graph in-degrees
    # Nodes with initial in-degree 0 must be tree nodes in the original graph structure (can't be on a cycle).
    # Removing these nodes and their outgoing edges might expose more nodes with in-degree 0, and so on.
    # The nodes remaining after this process must form the cycles.
    
    in_degree_orig = [0] * N
    for i in range(N):
        in_degree_orig[A[i]] += 1

    q_in_degree_zero = deque([i for i in range(N) if in_degree_orig[i] == 0])
    
    is_tree_node = [False] * N
    # tree_nodes_rev_topo stores tree nodes in an order suitable for DP (leaves of reversed trees first)
    tree_nodes_rev_topo = [] 

    # Use a copy of in_degree as we modify it
    current_in_degree = list(in_degree_orig)
    q_process = deque([i for i in range(N) if current_in_degree[i] == 0])
    
    # Use a set to keep track of nodes added to the topo queue to avoid duplicates
    added_to_topo_q = set(q_process)

    while q_process:
        u = q_process.popleft()
        is_tree_node[u] = True
        tree_nodes_rev_topo.append(u)
        
        v = A[u] # Original graph edge u -> v
        current_in_degree[v] -= 1
        
        # If reducing the in-degree of v makes it zero, and v is not already marked as tree node/processed
        # This condition might need refinement: if v's in-degree becomes 0, it means all incoming edges from
        # nodes ALREADY identified as tree nodes are processed. So v must be a tree node too.
        # It's important that we only add nodes to the queue if they haven't been processed yet.
        if current_in_degree[v] == 0 and not is_tree_node[v] and v not in added_to_topo_q:
             q_process.append(v)
             added_to_topo_q.add(v)

    # Nodes not marked as tree nodes are on cycles
    cycle_nodes_list = [i for i in range(N) if not is_tree_node[i]]

    # 3. Map tree nodes to cycles and identify cycles
    # BFS on the original graph starting from cycle nodes to find which cycle each tree node reaches.
    # This BFS essentially determines which tree nodes belong to which "component tree" rooted at a cycle node.

    node_to_cycle = [-1] * N # Stores the cycle index for each node (cycle node or tree node)
    cycles = [] # List of lists, each inner list is a cycle nodes

    # Find the distinct cycles and assign them IDs
    visited_cycle_finding = [False] * N
    
    for start_node in cycle_nodes_list:
        if not visited_cycle_finding[start_node]:
            # Find the cycle starting from start_node
            path = []
            curr = start_node
            # Follow path until a visited node is found
            while not visited_cycle_finding[curr]:
                visited_cycle_finding[curr] = True
                path.append(curr)
                curr = A[curr]

            # The first repeated node 'curr' is on the cycle. Find its first occurrence in path.
            cycle_start_index = path.index(curr)
            current_cycle = path[cycle_start_index:]
            
            cycle_id = len(cycles)
            cycles.append(current_cycle)
            
            # Assign cycle_id to all nodes in the current cycle
            for node in current_cycle:
                 node_to_cycle[node] = cycle_id

    K_prime = len(cycles) # Number of distinct cycles

    # Now, BFS from cycle nodes in the original graph to mark tree nodes with their cycle_id
    # Nodes are processed level by level based on distance FROM the cycle (in reversed graph).
    # This BFS uses original graph edges, essentially flowing information outwards from cycles.
    # If u -> v and v's cycle is known, u belongs to the same cycle's tree structure.
    
    q_bfs_cycle_map = deque(cycle_nodes_list) # Start BFS from all cycle nodes
    
    # The node_to_cycle for cycle nodes is already set. BFS expands to tree nodes.
    while q_bfs_cycle_map:
        u = q_bfs_cycle_map.popleft()
        current_cycle_id = node_to_cycle[u]

        # Iterate through nodes i such that A[i] = u (reversed graph edges)
        # If i is a tree node and hasn't been assigned a cycle yet, it belongs to the same cycle tree as u
        for i in rev_adj[u]:
            if is_tree_node[i] and node_to_cycle[i] == -1:
                node_to_cycle[i] = current_cycle_id
                q_bfs_cycle_map.append(i) # Add tree node to queue to explore its predecessors


    # Group tree nodes by the cycle they reach
    tree_nodes_by_cycle = [[] for _ in range(K_prime)]
    for i in range(N):
        if is_tree_node[i]:
            # node_to_cycle[i] was set by the BFS starting from cycles
            target_cycle_id = node_to_cycle[i]
            if target_cycle_id != -1: # Should be assigned a cycle if it's a tree node
                tree_nodes_by_cycle[target_cycle_id].append(i)


    # 4. Compute S[u][v] for all tree nodes u, v=1...M
    # S[u][v] is number of ways to assign values to reversed subtree rooted at u s.t. x_u <= v
    # This is done for tree nodes only.
    # The reversed topological sort order is in tree_nodes_rev_topo.
    # We need S[child][v] values to compute S[parent][v].

    # S table: S[u][v] for tree nodes u, v from 1 to M.
    # Initialize S values for tree nodes.
    S = {}
    for u in tree_nodes_rev_topo:
        S[u] = [0] * (M + 1) # S[u][0] is 0

    # Process tree nodes in reversed topological order (from leaves of reversed trees upwards)
    for u in tree_nodes_rev_topo:
        # Check if u is a leaf in the reversed graph among tree nodes
        # A tree node u is a leaf in the reversed tree structure rooted at a cycle if
        # no other tree node i has A[i] = u.
        
        is_rev_tree_leaf = True
        for i in rev_adj[u]:
            # Check if any node pointing to u (in reversed graph) is a tree node
            if is_tree_node[i]:
                is_rev_tree_leaf = False
                break

        if is_rev_tree_leaf:
            # Base case: if u is a leaf in the reversed tree of tree nodes,
            # constraints within its subtree are trivial. x_u can be 1 to v.
            for v in range(1, M + 1):
                S[u][v] = v
        else:
            # Recursive step: S[u][v] = S[u][v-1] + product of S[child][v] for children in reversed graph
            for v in range(1, M + 1):
                prod = 1
                # Children in reversed graph are nodes i such that A[i] = u
                # If u is a tree node, all such children i must also be tree nodes.
                for i in rev_adj[u]:
                     # S[i][v] for children i should have been computed already
                     # because i comes before u in reversed topological order
                     prod = (prod * S[i][v]) % MOD
                S[u][v] = (S[u][v-1] + prod) % MOD


    # 6. Compute final answer
    # Total ways = Product over cycles (Sum over values V for cycle (Product over tree nodes u reaching cycle (S[u][V])))
    total_ways = 1

    for j in range(K_prime):
        cycle_sum = 0 # Sum over possible values V for cycle j
        
        # For a fixed value V for cycle j
        for V in range(1, M + 1):
            prod_S_u_V = 1 # Product over tree nodes u that reach cycle j
            
            # tree_nodes_by_cycle[j] contains all tree nodes u that eventually reach cycle j
            for u in tree_nodes_by_cycle[j]:
                 # S[u][V] is the number of ways to assign values in the reversed subtree of u
                 # such that x_u <= V and internal constraints are met.
                 # Since u eventually reaches cycle j, the constraints on its value derive from
                 # the constraint chain u -> ... -> cycle node, and x_u <= x_{A_u} <= ... <= V.
                 # The calculation of S[u][v] already captures this.
                 prod_S_u_V = (prod_S_u_V * S[u][V]) % MOD

            # Add the count for this fixed cycle value V to the cycle_sum
            cycle_sum = (cycle_sum + prod_S_u_V) % MOD
        
        # Multiply the total ways by the sum for cycle j
        total_ways = (total_ways * cycle_sum) % MOD

    print(total_ways)

solve()