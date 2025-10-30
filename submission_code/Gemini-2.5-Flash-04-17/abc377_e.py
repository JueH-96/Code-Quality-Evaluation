# YOUR CODE HERE
import sys

# Use fast input reading
input = sys.stdin.readline

# Read N and K
N, K = map(int, input().split())

# Read the initial permutation P (1-based)
P = list(map(int, input().split()))

# Convert P to 0-based indexing for easier processing
# p_initial[i] represents the index that index i maps to in the initial permutation
p_initial = [p - 1 for p in P]

# The problem defines an operation on the permutation P.
# Let P^(k) be the permutation after k operations. P^(0) is the initial P.
# The operation is P^(k+1)_i = P^(k)_{P^(k)_i} for i=1, ..., N.
# If we represent the permutation P^(k) as a function g_k(i) = P^(k)_i,
# the operation means g_{k+1}(i) = g_k(g_k(i)) = (g_k o g_k)(i).
# Starting with g_0(i) = P_i, we get:
# g_1 = g_0 o g_0 = g_0^2
# g_2 = g_1 o g_1 = (g_0^2) o (g_0^2) = g_0^4
# g_3 = g_2 o g_2 = (g_0^4) o (g_0^4) = g_0^8
# After K operations, the permutation function is g_K = g_0^(2^K).
# We need to find the array of values P^(K)_i, which in 0-based indexing is aK[i] = g_K(i) = g_0^(2^K)(i).

# Array to store the final values (0-based)
aK = [0] * N

# Keep track of visited indices to find disjoint cycles of the permutation g_0
visited = [False] * N

# Iterate through each index from 0 to N-1
for i in range(N):
    # If index i has not been visited yet, it's the start of a new cycle
    if not visited[i]:
        # Start tracing the cycle from i
        cycle_nodes = []
        current = i
        
        # Follow the permutation g_0 path until we return to the starting node
        # Mark nodes as visited as we traverse them. Since p_initial is a permutation,
        # the traversal from any node will eventually lead back to itself, forming a cycle.
        while not visited[current]:
            visited[current] = True
            cycle_nodes.append(current)
            current = p_initial[current]

        # A cycle has been found. The nodes in the cycle are in the `cycle_nodes` list
        # in the order they are visited starting from i.
        # If cycle_nodes = [c_0, c_1, ..., c_{L-1}], then c_0 = i, c_1 = p_initial[c_0], ..., p_initial[c_{L-1}] = c_0.
        # This means g_0(c_j) = p_initial[c_j] = c_{(j+1)%L} where indices j are positions in the cycle_nodes list.
        L = len(cycle_nodes)
        
        # We need to compute g_K(idx) = g_0^(2^K)(idx) for each index idx in this cycle.
        # If an index is c_j (the node at position j in the cycle_nodes list),
        # applying g_0 m times maps c_j to c_{(j+m)%L}.
        # We need g_0^(2^K)(c_j), which maps c_j to c_{(j + 2^K)%L}.

        # Calculate the effective number of steps within the cycle: M_eff = 2^K mod L
        # We use Python's built-in modular exponentiation pow(base, exp, mod)
        # This handles large exponents efficiently.
        # pow(2, K, L) correctly computes (2**K) % L.
        # If L=1, pow(2, K, 1) = 0 for K >= 0. Taking 0 steps means the final value is the starting value.
        # If K=0, pow(2, 0, L) = 1 % L. Taking 1 step means the final value is g_0(original_value), which is the initial permutation.
        M_eff = pow(2, K, L)
        
        # Update the final permutation array aK for each node in the cycle
        # For each position j from 0 to L-1 in the cycle_nodes list:
        # The original index is current_cycle_idx = cycle_nodes[j].
        # The value at this index after K operations is g_K(current_cycle_idx) = g_0^(2^K)(current_cycle_idx).
        # This value is the index obtained by taking M_eff steps along the cycle starting from current_cycle_idx.
        # Since current_cycle_idx is at position j in cycle_nodes, M_eff steps away is the index at position (j + M_eff)%L in cycle_nodes.
        # The index itself is target_cycle_idx = cycle_nodes[(j + M_eff) % L].
        # The value at index current_cycle_idx in the final array aK is target_cycle_idx.
        for j in range(L):
            current_cycle_idx = cycle_nodes[j]
            target_cycle_idx = cycle_nodes[(j + M_eff) % L]
            aK[current_cycle_idx] = target_cycle_idx

# Print the final permutation (1-based values), separated by spaces
print(*(val + 1 for val in aK))