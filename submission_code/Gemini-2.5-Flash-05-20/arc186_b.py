import sys

# Set recursion limit for deep DFS, as N can be up to 3*10^5
sys.setrecursionlimit(3 * 10**5 + 100)

MOD = 998244353

def solve():
    N = int(sys.stdin.readline())
    # A is 0-indexed in input, corresponding to P_1 to P_N.
    # So A[i] is the value for P_{i+1}.
    A = list(map(int, sys.stdin.readline().split()))

    # Build the tree structure where A[k] is the parent of node k+1.
    # We use 1-based indexing for nodes (1 to N) to match P_i.
    # Node 0 is a conceptual parent for elements whose A value is 0.
    adj = [[] for _ in range(N + 1)] 
    
    # Store indices of nodes that are children of conceptual node 0 (i.e., A[k-1] == 0).
    # These are the roots of the forest structure defined by A.
    roots_of_forest = []

    for i in range(N):
        parent_idx = A[i]       # Parent index from input A (0-indexed A[i] is parent of node i+1)
        child_idx = i + 1        # Current node (1-indexed)
        adj[parent_idx].append(child_idx)
        if parent_idx == 0:
            roots_of_forest.append(child_idx)
            
    # Precompute factorials and inverse factorials modulo MOD.
    # fact[k] = k! % MOD
    # inv_fact[k] = (k!)^(-1) % MOD
    fact = [1] * (N + 1)
    inv_fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i-1] * i) % MOD
    
    # Modular inverse for factorials using Fermat's Little Theorem: a^(MOD-2) % MOD
    inv_fact[N] = pow(fact[N], MOD - 2, MOD)
    for i in range(N - 1, -1, -1): # Iterate down to 0, inv_fact[0] is 1
        inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD

    # sz[u] will store the size of the subtree rooted at u (including u itself).
    # sz[0] will store the total count of nodes in the entire forest + 1 (for conceptual root 0 itself).
    sz = [0] * (N + 1)
    
    # DFS function to compute subtree sizes.
    # This DFS starts from the conceptual root 0 and computes sz for all actual nodes 1 to N.
    def dfs_compute_subtree_sizes(u):
        sz[u] = 1 # Initialize size with the node u itself
        for v in adj[u]: # Iterate through children of u
            dfs_compute_subtree_sizes(v) # Recursively compute size for child subtree
            sz[u] += sz[v] # Add child's subtree size to u's subtree size

    # Start DFS from conceptual root 0. This will correctly populate sz for all nodes from 1 to N.
    dfs_compute_subtree_sizes(0) 

    # Calculate the product of all subtree sizes for nodes 1 to N.
    # This forms part of the denominator in the final formula.
    total_sz_product = 1
    for i in range(1, N + 1): # Iterate for nodes 1 to N (P_1 to P_N)
        total_sz_product = (total_sz_product * sz[i]) % MOD
    
    # The base answer is N! / (product of all subtree sizes).
    # This is calculated using modular inverse for division.
    ans = (fact[N] * pow(total_sz_product, MOD - 2, MOD)) % MOD
    
    # Adjust the answer based on the number of roots in the forest.
    # As derived from sample and problem conditions, if there are `m` roots
    # (nodes `i` where `A_{i-1}=0`), the answer is divided by `2^(m-1)`.
    num_roots = len(roots_of_forest)
    if num_roots > 1:
        # Calculate 2^(num_roots - 1)
        two_power_factor = pow(2, num_roots - 1, MOD)
        # Divide `ans` by `two_power_factor` by multiplying with its modular inverse.
        ans = (ans * pow(two_power_factor, MOD - 2, MOD)) % MOD

    sys.stdout.write(str(ans) + "
")

solve()