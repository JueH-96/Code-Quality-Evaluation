import sys

# Increase recursion limit for deep DFS paths in the component tree.
# Max depth can be N. For N=2e5, set limit higher.
sys.setrecursionlimit(4 * 10**5 + 50) 
MOD = 998244353

def power(a, b):
    res = 1
    a %= MOD
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % MOD
        a = (a * a) % MOD
        b //= 2
    return res

def inv(n):
    return power(n, MOD - 2)

# Globals for DSU and Component Tree structures for convenience in CP.
# DSU data structures
parent_dsu = [] 
comp_node_id = [] # For a DSU root i, comp_node_id[i] is its current component tree node ID
team_sz = []      # For a DSU root i, team_sz[i] is its current team size

# Component Tree data structures
node_val = [] # Stores sums of probabilities for wins. Indexed by component tree node ID.
children = [] # Adjacency list for component tree. children[u] = list of children of u.

# Final answers
player_ans = []
N_global = 0 # To pass N to DFS if needed (e.g. to identify leaves)

def find_set(player_id_): # Use a distinct name for parameter
    # Path compression
    if parent_dsu[player_id_] == player_id_:
        return player_id_
    parent_dsu[player_id_] = find_set(parent_dsu[player_id_])
    return parent_dsu[player_id_]

def dfs_calculate_ans(u_comp_node, current_sum_from_ancestors):
    # val_at_u is sum of probabilities for node u and its ancestors in component tree
    val_at_u = (current_sum_from_ancestors + node_val[u_comp_node]) % MOD
    
    # If u_comp_node is a leaf (represents an original player)
    if u_comp_node <= N_global: 
        player_ans[u_comp_node] = val_at_u
    
    # Recursively call for children
    for v_child_comp_node in children[u_comp_node]:
        dfs_calculate_ans(v_child_comp_node, val_at_u)

def solve():
    global N_global, parent_dsu, comp_node_id, team_sz, node_val, children, player_ans

    N = int(sys.stdin.readline())
    N_global = N # Store N for DFS to identify leaf nodes

    # Initialize DSU structures (1-indexed for player IDs 1 to N)
    parent_dsu = [0] * (N + 1)
    comp_node_id = [0] * (N + 1) 
    team_sz = [0] * (N + 1)      

    for i in range(1, N + 1):
        parent_dsu[i] = i
        comp_node_id[i] = i # Player i initially corresponds to component tree node i
        team_sz[i] = 1

    # Component Tree nodes 1..N are leaves (players). Nodes N+1..2N-1 are internal.
    # Max node ID is 2*N-1. Size 2*N for 1-indexing up to 2N-1.
    num_comp_tree_nodes = 2 * N 
    node_val = [0] * num_comp_tree_nodes
    
    # Store parent pointers for component tree nodes to rebuild adjacency list later
    comp_tree_node_actual_parents = [0] * num_comp_tree_nodes 

    curr_internal_node_idx = N + 1 # Next available ID for internal component tree nodes

    matches_input = []
    for _ in range(N - 1): # There are N-1 matches
        u, v = map(int, sys.stdin.readline().split())
        matches_input.append((u,v))

    for p_player, q_player in matches_input:
        r_p = find_set(p_player) # DSU root for p_player's team
        r_q = find_set(q_player) # DSU root for q_player's team
        
        # comp_node_p_id and comp_node_q_id are the component tree nodes
        # representing the teams of p_player and q_player respectively BEFORE this match's merge
        comp_node_p_id = comp_node_id[r_p] 
        comp_node_q_id = comp_node_id[r_q] 

        s_p = team_sz[r_p] # Size of p_player's team
        s_q = team_sz[r_q] # Size of q_player's team

        inv_sum_sizes = inv(s_p + s_q)
        
        prob_p_wins = (s_p * inv_sum_sizes) % MOD
        prob_q_wins = (s_q * inv_sum_sizes) % MOD

        # Accumulate probabilities at the component tree nodes representing the teams playing
        node_val[comp_node_p_id] = (node_val[comp_node_p_id] + prob_p_wins) % MOD
        node_val[comp_node_q_id] = (node_val[comp_node_q_id] + prob_q_wins) % MOD
        
        # Create new component tree node for the merged team
        new_merged_comp_node = curr_internal_node_idx
        curr_internal_node_idx += 1

        # Set parent pointers in the component tree structure
        comp_tree_node_actual_parents[comp_node_p_id] = new_merged_comp_node
        comp_tree_node_actual_parents[comp_node_q_id] = new_merged_comp_node

        # DSU union operation (using union by size heuristic)
        # The DSU root of the larger set becomes the parent of the smaller set's root.
        # The new DSU root (parent) gets its component_node_id updated to new_merged_comp_node.
        if team_sz[r_p] < team_sz[r_q]:
            parent_dsu[r_p] = r_q
            team_sz[r_q] = s_p + s_q
            comp_node_id[r_q] = new_merged_comp_node # r_q is new root, update its comp_node_id
        elif team_sz[r_q] < team_sz[r_p]:
            parent_dsu[r_q] = r_p
            team_sz[r_p] = s_p + s_q
            comp_node_id[r_p] = new_merged_comp_node # r_p is new root
        else: # Sizes are equal, merge r_q into r_p (can also use rank)
            parent_dsu[r_q] = r_p
            team_sz[r_p] = s_p + s_q
            comp_node_id[r_p] = new_merged_comp_node # r_p is new root
            
    # Rebuild adjacency list (children) for component tree from parent pointers
    children = [[] for _ in range(num_comp_tree_nodes)]
    
    # The root of the component tree is the last 'new_merged_comp_node' created.
    # For N >= 2, there are N-1 matches, so N-1 internal nodes N+1, ..., 2N-1.
    # The last index used is 2N-1.
    comp_tree_root = (2 * N - 1) if N > 1 else 1 # if N=1, root is node 1. Problem says N>=2.

    # Iterate all component tree nodes (1 to 2N-1) to build children lists
    # Nodes are 1 to (curr_internal_node_idx - 1)
    for i in range(1, curr_internal_node_idx): 
        parent_of_i = comp_tree_node_actual_parents[i]
        if parent_of_i != 0: # If i is not the root of the component tree
            children[parent_of_i].append(i)
    
    player_ans = [0] * (N + 1) # 1-indexed answers
    
    # Perform DFS from component tree root to calculate final expected values
    # Initial sum from ancestors is 0 for the root.
    dfs_calculate_ans(comp_tree_root, 0)

    # Print results
    result_strs = [str(player_ans[i]) for i in range(1, N + 1)]
    sys.stdout.write(" ".join(result_strs) + "
")

solve()