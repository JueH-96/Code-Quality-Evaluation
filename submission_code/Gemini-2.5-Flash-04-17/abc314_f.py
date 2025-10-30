import sys

# Increase recursion depth for DFS
# Adjust this based on typical contest limits or analysis of merge structure depth
# Maximum depth could be N-1 in a skewed tree
sys.setrecursionlimit(300000)

MOD = 998244353

# Global variables for DSU and tree construction
dsu_parent = []
dsu_sz = []
# Map DSU root player index (1..N_players) to tree node index (1..2*N_players-1)
node_map = []
# Adjacency list for tree structure. Index is node ID - 1.
tree_children = []
# Stores parent node ID. Index is node ID - 1.
tree_parent = []
# Stores probability that the node's team won against its sibling to form the parent team. Index is node ID - 1.
match_win_prob = []

def mod_inverse(a, m):
    """Compute modular multiplicative inverse using Fermat's Little Theorem."""
    return pow(a, m - 2, m)

def find_dsu(i):
    """Find root of the DSU set containing player i with path compression."""
    if dsu_parent[i] == i:
        return i
    dsu_parent[i] = find_dsu(dsu_parent[i])
    return dsu_parent[i]

def union_dsu(p, q, current_node_idx):
    """Union DSU sets for players p and q, build merge tree node."""
    global tree_children, tree_parent, match_win_prob, node_map, dsu_parent, dsu_sz, MOD

    root_p = find_dsu(p)
    root_q = find_dsu(q)

    if root_p == root_q:
        # This case should not happen based on problem constraints
        return root_p

    sz_p = dsu_sz[root_p]
    sz_q = dsu_sz[root_q]

    inv_sum = mod_inverse(sz_p + sz_q, MOD)
    prob_p_wins = (sz_p * inv_sum) % MOD
    prob_q_wins = (sz_q * inv_sum) % MOD

    # Create new internal node in the merge tree
    new_node_id = current_node_idx[0]
    current_node_idx[0] += 1

    node_p_id = node_map[root_p]
    node_q_id = node_map[root_q]

    # Store tree structure using node IDs. Map ID to list index by subtracting 1.
    tree_parent[node_p_id - 1] = new_node_id
    tree_parent[node_q_id - 1] = new_node_id
    tree_children[new_node_id - 1].append(node_p_id)
    tree_children[new_node_id - 1].append(node_q_id)
    match_win_prob[node_p_id - 1] = prob_p_wins
    match_win_prob[node_q_id - 1] = prob_q_wins


    # Union DSU sets by size (merge smaller set into larger set)
    if dsu_sz[root_p] < dsu_sz[root_q]:
        root_p, root_q = root_q, root_p # Swap roles to merge q into p

    dsu_parent[root_q] = root_p
    dsu_sz[root_p] += dsu_sz[root_q]

    # Update node_map: the new DSU root (root_p) now maps to the new tree node (new_node_id)
    node_map[root_p] = new_node_id

    return root_p # Return the new DSU root

# Expected wins array (1-based indexing for players 1..N_players)
E = []
N_players = 0 # Global variable for number of players

# DFS to compute expected wins for each player
# u_node_id is the ID of the current node being visited (1..2*N_players-1)
# current_sum_prob is the sum of win probabilities on the path from u_node_id's parent up to the root
def dfs_expected_wins(u_node_id, current_sum_prob):
    global E, N_players, tree_children, match_win_prob

    if u_node_id <= N_players: # Base case: If the node ID is <= N_players, it's a leaf node representing a player
        E[u_node_id] = current_sum_prob
        return

    # Map node_id (1..2*N_players-1) to list index (0..2*N_players-2)
    u_list_idx = u_node_id - 1

    # Recursively call DFS for children nodes
    for c_node_id in tree_children[u_list_idx]:
        c_list_idx = c_node_id - 1
        # The probability associated with the edge from child c to parent u is match_win_prob[c_list_idx]
        prob_c_wins_match = match_win_prob[c_list_idx]
        
        # The sum of probabilities for the path from this child c up to the root is
        # (sum from u's parent to root) + (prob c wins against sibling to form u)
        next_sum_prob = (current_sum_prob + prob_c_wins_match) % MOD
        
        dfs_expected_wins(c_node_id, next_sum_prob)


# Main execution function
def solve():
    global N_players, dsu_parent, dsu_sz, node_map, tree_children, tree_parent, match_win_prob, E

    N_players = int(sys.stdin.readline())
    matches = []
    for _ in range(N_players - 1):
        p, q = map(int, sys.stdin.readline().split())
        matches.append((p, q))

    # Initialize DSU structure (1-based indexing for players 1..N_players)
    dsu_parent = list(range(N_players + 1))
    dsu_sz = [1] * (N_players + 1)
    # Initialize node_map: initially player i (1..N_players) maps to tree node i (1..N_players)
    node_map = list(range(N_players + 1))

    # Initialize tree structure arrays
    # Total nodes in the merge tree are 2*N_players - 1 (N leaves + N-1 internal nodes)
    # We use 1-based indexing for node IDs (1 to 2*N_players-1).
    # Array size 2*N_players is sufficient for indices 0 to 2*N_players-1.
    max_nodes = 2 * N_players
    tree_children = [[] for _ in range(max_nodes)]
    tree_parent = [0] * max_nodes
    match_win_prob = [0] * max_nodes

    # current_node_idx_val generates IDs for internal nodes, starting from N_players + 1
    # These IDs will go up to N_players + (N_players - 1) = 2*N_players - 1
    current_node_idx_val = [N_players + 1] # Use list for mutable integer

    # Process matches and build the merge tree by performing DSU unions
    last_dsu_root = None
    for p, q in matches:
        last_dsu_root = union_dsu(p, q, current_node_idx_val)

    # The root node of the merge tree is the node mapped by the final DSU root
    # Any player's find_dsu result will give the final DSU root after all merges
    final_dsu_root = find_dsu(1)
    root_node_id = node_map[final_dsu_root] # root_node_id is an ID in 1..2*N_players-1

    # Initialize Expected wins array (1-based indexing for players 1..N_players)
    E = [0] * (N_players + 1)

    # Perform DFS traversal starting from the root of the merge tree
    # The initial current_sum_prob is 0 for the root
    dfs_expected_wins(root_node_id, 0)

    # Print the results for players 1 to N_players
    print(*E[1:])

# Execute the solve function
solve()