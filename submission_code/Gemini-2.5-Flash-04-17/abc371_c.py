import sys
import itertools

# Function to convert 1-indexed vertices to 0-indexed
def to_0_indexed(v):
    return v - 1

# Read input
N = int(sys.stdin.readline())

# Read graph G
M_G = int(sys.stdin.readline())
g_adj = [[False] * N for _ in range(N)]
for _ in range(M_G):
    u, v = map(int, sys.stdin.readline().split())
    u_0 = to_0_indexed(u)
    v_0 = to_0_indexed(v)
    g_adj[u_0][v_0] = g_adj[v_0][u_0] = True

# Read graph H
M_H = int(sys.stdin.readline())
h_adj = [[False] * N for _ in range(N)]
for _ in range(M_H):
    a, b = map(int, sys.stdin.readline().split())
    a_0 = to_0_indexed(a)
    b_0 = to_0_indexed(b)
    h_adj[a_0][b_0] = h_adj[b_0][a_0] = True

# Read costs A_i,j
a_cost = [[0] * N for _ in range(N)]
# The costs are given for pairs (i, j) with 1 <= i < j <= N
# In 0-indexing, this corresponds to pairs (i', j') with 0 <= i' < j' <= N-1
# The input format lists costs row by row for i=1, then i=2, ..., i=N-1
# For a fixed i (1-indexed), costs A_{i,j} are listed for j = i+1, i+2, ..., N
# In 0-indexing, for a fixed i' (0-indexed), costs A_{i'+1, j'+1} are listed for j' = i'+1, i'+2, ..., N-1
for i in range(N - 1): # i runs from 0 to N-2 (representing 1-indexed vertices 1 to N-1)
    # Read the line of costs for the current i
    row_costs = list(map(int, sys.stdin.readline().split()))
    
    # These costs are for pairs (i, j) where j is greater than i
    # The indices in the `row_costs` list correspond to j = i + 1, i + 2, ..., N - 1
    # The first cost in `row_costs` is for j = i + 1, the second for j = i + 2, and so on.
    for j_idx in range(len(row_costs)):
        j = i + 1 + j_idx # Calculate the actual j (0-indexed) corresponding to the current cost
        # The cost A_{i+1, j+1} (1-indexed) is A_cost[i][j] (0-indexed)
        a_cost[i][j] = a_cost[j][i] = row_costs[j_idx]

# Iterate through all permutations of vertices 0 to N-1
# A permutation p = (p_0, p_1, ..., p_{N-1}) means vertex i in G maps to vertex p[i] in H.
min_total_cost = float('inf')

# Handle N=1 case: The loops below naturally result in cost 0, which is correct.
# For N=1, itertools.permutations(range(1)) yields ((0),). The inner loops for v1, v2 won't run.

for p in itertools.permutations(range(N)):
    # p is a tuple representing the mapping from G vertex index (0 to N-1) to H vertex index (0 to N-1)
    # G vertex i maps to H vertex p[i]
    
    # Construct the inverse permutation p_inv
    # p_inv[v] will give the vertex i in G such that p[i] = v (i.e., G vertex i maps to H vertex v)
    p_inv = [0] * N
    for i in range(N):
        p_inv[p[i]] = i

    current_cost = 0
    
    # Iterate through all possible edges (v1, v2) in H (0-indexed)
    # We only need to check each pair once, typically by ensuring v1 < v2
    for v1 in range(N):
        for v2 in range(v1 + 1, N):
            # Consider the edge between vertices v1 and v2 in H.
            # Under the permutation p, vertex v1 in H corresponds to vertex p_inv[v1] in G,
            # and vertex v2 in H corresponds to vertex p_inv[v2] in G.
            u1 = p_inv[v1]
            u2 = p_inv[v2]

            # For H to be isomorphic to G under permutation p, the edge (v1, v2) must exist in H
            # if and only if the edge (u1, u2) exists in G.
            target_h_has_edge = g_adj[u1][u2]
            
            # The current state of the edge (v1, v2) in H
            current_h_has_edge = h_adj[v1][v2]

            # If the current state of the edge (v1, v2) in H does not match the required state
            # based on the corresponding edge (u1, u2) in G, we need to flip the edge (v1, v2) in H.
            # The cost of this flip is A_cost[v1][v2].
            if current_h_has_edge != target_h_has_edge:
                current_cost += a_cost[v1][v2]
                
    # Update the minimum cost found so far across all permutations
    min_total_cost = min(min_total_cost, current_cost)

print(min_total_cost)