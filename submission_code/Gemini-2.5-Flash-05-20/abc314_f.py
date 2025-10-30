import sys

# Set recursion limit for deep trees (N-1 matches can form a "linked list" like tree structure)
sys.setrecursionlimit(2 * 10**5 + 100) 

# Modulo constant
MOD = 998244353

# Modular exponentiation for a^b % MOD
def fast_pow(base, power):
    result = 1
    base %= MOD
    while power > 0:
        if power % 2 == 1:
            result = (result * base) % MOD
        base = (base * base) % MOD
        power //= 2
    return result

# Modular inverse using Fermat's Little Theorem (a^(MOD-2) % MOD for prime MOD)
def mod_inverse(n):
    return fast_pow(n, MOD - 2)

class DSU:
    def __init__(self, n):
        # parent[i]: parent of element i
        self.parent = list(range(n + 1))
        # size[i]: size of the set if i is the root
        self.size = [1] * (n + 1)
        # dsu_root_to_tree_node[i]: maps DSU root i to its corresponding node ID in the explicit tournament tree
        # Initially, each player i is its own DSU root and corresponds to tree node i
        self.dsu_root_to_tree_node = list(range(n + 1))

    # Finds the representative (root) of the set containing element i, with path compression
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

# DFS to calculate expected wins for each player
def dfs_calculate_expectations(node, accumulated_prob, tree_adj, team_component_size, final_expectations, N_players):
    # If it's a leaf node (original player ID)
    if node <= N_players: 
        final_expectations[node] = accumulated_prob
        return

    # It's an internal node, representing a merge of two teams
    # Get its children (the two teams that merged in this match)
    child1, child2 = tree_adj[node]

    # Get the sizes of the two merging teams (at the time of their merge)
    size1 = team_component_size[child1]
    size2 = team_component_size[child2]

    # Calculate win probabilities for this particular match
    denom = (size1 + size2) % MOD
    inv_denom = mod_inverse(denom)

    prob1_wins = (size1 * inv_denom) % MOD # Probability child1's team wins this match
    prob2_wins = (size2 * inv_denom) % MOD # Probability child2's team wins this match

    # Recurse for children. The accumulated probability for a child's player includes
    # the probability that their team won this current match.
    dfs_calculate_expectations(child1, (accumulated_prob + prob1_wins) % MOD, tree_adj, team_component_size, final_expectations, N_players)
    dfs_calculate_expectations(child2, (accumulated_prob + prob2_wins) % MOD, tree_adj, team_component_size, final_expectations, N_players)

def solve():
    N = int(sys.stdin.readline())
    matches = []
    for _ in range(N - 1):
        p, q = map(int, sys.stdin.readline().split())
        matches.append((p, q))

    dsu = DSU(N)

    # tree_adj: Adjacency list for the explicit tournament tree.
    # Nodes 1 to N are players. Nodes N+1 to 2N-1 are internal nodes representing merged teams.
    # Max node ID is 2N-1. Allocate space up to 2N. (Using 1-based indexing for player IDs and tree nodes)
    tree_adj = [[] for _ in range(2 * N)]

    # team_component_size[i]: Stores the total number of players in the team represented by node i.
    # For players (leaves, nodes 1 to N), it's 1. For internal nodes (merged teams), it's the sum of its children's sizes.
    team_component_size = [0] * (2 * N)
    # Initialize for player nodes (leaves)
    for i in range(1, N + 1):
        team_component_size[i] = 1

    # node_id_counter: Used to assign unique IDs to new internal nodes (merged teams), starting from N+1
    node_id_counter = N + 1 

    # Simulate matches and build the explicit tournament tree
    # overall_dsu_root will track the DSU root of the current largest component.
    # After N-1 matches, this will be the root of the single team containing all players.
    overall_dsu_root = -1 

    for p, q in matches:
        root_p = dsu.find(p)
        root_q = dsu.find(q)

        # Get the current sizes of the teams (DSU component sizes)
        s_p = dsu.size[root_p]
        s_q = dsu.size[root_q]
        
        # Get the explicit tree node IDs corresponding to these DSU roots BEFORE they merge
        tree_node_p = dsu.dsu_root_to_tree_node[root_p]
        tree_node_q = dsu.dsu_root_to_tree_node[root_q]

        # Assign a new tree node ID for the merged team
        new_tree_node_id = node_id_counter
        node_id_counter += 1

        # Add the two merging teams as children to the new merged team node
        tree_adj[new_tree_node_id].append(tree_node_p)
        tree_adj[new_tree_node_id].append(tree_node_q)
        # The size of the new merged team is the sum of its children's sizes
        team_component_size[new_tree_node_id] = s_p + s_q

        # Perform DSU union (union by size)
        # Make the root of the smaller DSU tree point to the root of the larger DSU tree
        if dsu.size[root_p] < dsu.size[root_q]:
            dsu.parent[root_p] = root_q
            dsu.size[root_q] += dsu.size[root_p]
            # Update the mapping for the new DSU root to point to the newly created tree node
            dsu.dsu_root_to_tree_node[root_q] = new_tree_node_id
            overall_dsu_root = root_q # Update the current overall DSU root
        else:
            dsu.parent[root_q] = root_p
            dsu.size[root_p] += dsu.size[root_q]
            dsu.dsu_root_to_tree_node[root_p] = new_tree_node_id
            overall_dsu_root = root_p # Update the current overall DSU root

    # The root of the overall explicit tournament tree is the tree node ID corresponding to the final DSU root.
    # We can find this by finding the root of any player (e.g., player 1) in the DSU at the end.
    overall_tree_root = dsu.dsu_root_to_tree_node[dsu.find(1)]

    # final_expectations[i] will store the expected number of wins for player i's team
    final_expectations = [0] * (N + 1)
    # Start the DFS from the overall root, with initial accumulated probability of 0
    dfs_calculate_expectations(overall_tree_root, 0, tree_adj, team_component_size, final_expectations, N)

    # Print the results for each player from 1 to N
    print(*(final_expectations[i] for i in range(1, N + 1)))

solve()