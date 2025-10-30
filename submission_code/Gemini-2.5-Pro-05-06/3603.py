class Solution:
  def findAnswer(self, parent: list[int], s: str) -> list[bool]:
    n = len(parent)

    if n == 0: # Based on constraints (n >= 1), this case won't occur.
        return []

    adj = [[] for _ in range(n)]
    # parent[0] is -1, so node 0 is the root.
    # Children are added to adjacency lists. Since we iterate i from 1 to n-1,
    # for any parent p, its children parent[i]=p will be added to adj[p]
    # in increasing order of i. So, adj[p] will be sorted by child index.
    for i in range(1, n): # parent[0] is -1, so loop starts from 1
      if parent[i] != -1: # Should always be true for i > 0 in a valid tree
        adj[parent[i]].append(i)

    # Hash parameters (using two hash functions for robustness)
    P1, MOD1 = 31, 10**9 + 7
    P2, MOD2 = 37, 10**9 + 9 # Another prime base and large prime modulus

    # Precompute powers of P1 and P2
    # POW1[k] = P1^k % MOD1. Max power needed is N-1 for a string of length N.
    # An array of size N (indices 0 to N-1) for P^0 to P^(N-1) is enough.
    # Using size N+1 for safety/simplicity to store up to P^N.
    POW1 = [1] * (n + 1) 
    POW2 = [1] * (n + 1)
    for i in range(1, n + 1):
      POW1[i] = (POW1[i-1] * P1) % MOD1
      POW2[i] = (POW2[i-1] * P2) % MOD2

    # Arrays to store hash values and lengths for S(u)
    h1_vals = [0] * n  # Forward hash with (P1, MOD1)
    hr1_vals = [0] * n # Reverse hash with (P1, MOD1)
    h2_vals = [0] * n  # Forward hash with (P2, MOD2)
    hr2_vals = [0] * n # Reverse hash with (P2, MOD2)
    lengths = [0] * n  # Length of the string S(u)
    
    answer = [False] * n # Stores the boolean result for each node

    # Iterative post-order traversal
    # Root of the tree is 0.
    stack1_dfs = [0] # Start DFS from root 0
    stack2_post_order_reversed = []

    while stack1_dfs:
        u = stack1_dfs.pop()
        stack2_post_order_reversed.append(u)
        # Children are already sorted in adj[u]
        # To process children c1, c2, ..., ck in order for post-order,
        # they should be pushed onto stack1 as ck, ..., c2, c1.
        # Here, adj[u] is [c1, c2, ..., ck]. Iterating through it normally
        # and pushing to stack1 means c1 is pushed, then c2, ..., then ck.
        # So ck is at the top of stack1 and processed first among siblings.
        # This order works for the two-stack post-order generation method.
        for v_child in adj[u]: 
            stack1_dfs.append(v_child)
            
    post_order_nodes = stack2_post_order_reversed[::-1] # Actual post-order

    # Calculate hashes for each node in post-order
    for u_node in post_order_nodes:
        # Character value for s[u_node] (1-indexed for 'a' to 'z')
        val_s_u = ord(s[u_node]) - ord('a') + 1
        
        # Initialize aggregates for children's concatenated string part
        child_concat_h1, child_concat_hr1 = 0, 0
        child_concat_h2, child_concat_hr2 = 0, 0
        child_concat_len = 0

        # Process children (which are already processed due to post-order traversal)
        for v_child in adj[u_node]: # Children are already sorted by index
            # Accumulate forward hash for S_children = S_children_so_far + S(v_child)
            # H_f(A B) = H_f(A) * P^{len(B)} + H_f(B)
            child_concat_h1 = (child_concat_h1 * POW1[lengths[v_child]] + h1_vals[v_child]) % MOD1
            child_concat_h2 = (child_concat_h2 * POW2[lengths[v_child]] + h2_vals[v_child]) % MOD2
            
            # Accumulate reverse hash for S_children = S_children_so_far + S(v_child)
            # H_r(A B) = H_r(A) + H_r(B) * P^{len(A)}
            child_concat_hr1 = (child_concat_hr1 + hr1_vals[v_child] * POW1[child_concat_len]) % MOD1
            child_concat_hr2 = (child_concat_hr2 + hr2_vals[v_child] * POW2[child_concat_len]) % MOD2
            
            child_concat_len += lengths[v_child]

        # Finalize hashes for S(u_node) = S(children_concat) + s[u_node]
        # H_f(S(u_node)) = H_f(S_children_concat) * P^1 + val(s[u_node])
        h1_vals[u_node] = (child_concat_h1 * POW1[1] + val_s_u) % MOD1
        h2_vals[u_node] = (child_concat_h2 * POW2[1] + val_s_u) % MOD2
        
        # H_r(S(u_node)) = H_r(S_children_concat) + val(s[u_node]) * P^{len(S_children_concat)}
        hr1_vals[u_node] = (child_concat_hr1 + val_s_u * POW1[child_concat_len]) % MOD1
        hr2_vals[u_node] = (child_concat_hr2 + val_s_u * POW2[child_concat_len]) % MOD2
        
        lengths[u_node] = child_concat_len + 1
        
        # Check for palindrome property
        answer[u_node] = (h1_vals[u_node] == hr1_vals[u_node] and \
                          h2_vals[u_node] == hr2_vals[u_node])

    return answer