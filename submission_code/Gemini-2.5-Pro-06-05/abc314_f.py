# YOUR CODE HERE
import sys

def solve_problem():
    """
    This function contains the full solution logic.
    It's wrapped in a function to avoid global scope issues.
    """

    # Set a higher recursion limit for deep merge trees.
    # A skewed tree can have a depth of N-1.
    # N <= 2 * 10^5, so a limit of 2*N is very safe.
    sys.setrecursionlimit(4 * 10**5 + 50) 

    # The modulus for calculations
    MOD = 998244353

    def power(a, b):
        """
        Calculates (a^b) % MOD using binary exponentiation.
        """
        res = 1
        a %= MOD
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % MOD
            a = (a * a) % MOD
            b //= 2
        return res

    def modInverse(n):
        """
        Calculates modular inverse of n modulo MOD using Fermat's Little Theorem.
        Assumes MOD is a prime number.
        """
        return power(n, MOD - 2)

    # Using a dictionary for memoization to speed up repeated inverse calculations.
    memo_inv = {}
    def fast_mod_inverse(n):
        if n not in memo_inv:
            memo_inv[n] = modInverse(n)
        return memo_inv[n]

    # Fast I/O
    input = sys.stdin.readline
    
    try:
        n_str = input()
        if not n_str: return
        N = int(n_str)
    except (IOError, ValueError):
        return

    # A single player has 0 expected wins as there are no matches.
    if N == 1:
        print(0)
        return

    matches = []
    for _ in range(N - 1):
        matches.append(list(map(int, input().split())))

    # --- Step 1: Build the merge tree using a DSU ---
    
    # DSU data structures (1-based indexing for players)
    dsu_parent = list(range(N + 1))
    dsu_size = [1] * (N + 1)

    def find(i):
        # Path compression for DSU
        if dsu_parent[i] == i:
            return i
        dsu_parent[i] = find(dsu_parent[i])
        return dsu_parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        # The problem guarantees p_i and q_i are in different teams.
        
        # Union by size for DSU
        if dsu_size[root_i] < dsu_size[root_j]:
            root_i, root_j = root_j, root_i
        dsu_parent[root_j] = root_i
        dsu_size[root_i] += dsu_size[root_j]
        return root_i

    # Merge tree data structures
    # Nodes 1..N are players (leaves)
    # Nodes N+1..2N-1 are internal merge nodes
    num_nodes = 2 * N 
    adj = [[] for _ in range(num_nodes)]
    team_size = [0] * num_nodes
    
    # Initialize leaf nodes (players)
    for i in range(1, N + 1):
        team_size[i] = 1

    # `component_node` maps a DSU component's root to its corresponding node in the merge tree.
    component_node = list(range(N + 1))
    
    next_node_id = N + 1

    for p, q in matches:
        root_p = find(p)
        root_q = find(q)

        node_p = component_node[root_p]
        node_q = component_node[root_q]
        
        # Create a new internal node for the merge
        new_node = next_node_id
        next_node_id += 1
        
        # The new node is the parent of the two merging nodes
        adj[new_node].append(node_p)
        adj[new_node].append(node_q)
        
        # The size of the new team is the sum of the sizes of the merging teams
        team_size[new_node] = team_size[node_p] + team_size[node_q]
        
        # Merge the components in the DSU
        new_dsu_root = union(p, q)
        # Update the map to point the new component's root to the new tree node
        component_node[new_dsu_root] = new_node

    # The root of the entire merge tree is the last node created
    root = next_node_id - 1
    
    # --- Step 2: Calculate expected values using a top-down DFS on the tree ---
    
    ans = [0] * (N + 1)
    
    def dfs(v, E_so_far):
        """
        Performs a DFS traversal from the root of the merge tree.
        `v`: current node in the tree.
        `E_so_far`: the accumulated expected wins from merges at ancestor nodes.
        """
        # Base case: if `v` is a leaf node (a player), we have the final answer.
        if v <= N:
            ans[v] = E_so_far
            return

        # `v` is an internal node, representing a match.
        u1, u2 = adj[v]
        s1 = team_size[u1]
        s2 = team_size[u2]
        
        total_size = s1 + s2
        inv_total_size = fast_mod_inverse(total_size)
        
        # Probability that team `u1` wins this match
        prob1_wins = (s1 * inv_total_size) % MOD
        # Probability that team `u2` wins this match
        prob2_wins = (s2 * inv_total_size) % MOD
        
        # Recurse for children, adding the win probability for their respective teams.
        dfs(u1, (E_so_far + prob1_wins) % MOD)
        dfs(u2, (E_so_far + prob2_wins) % MOD)

    dfs(root, 0)
    
    # Print the results for players 1 to N
    print(*ans[1:])

if __name__ == "__main__":
    solve_problem()