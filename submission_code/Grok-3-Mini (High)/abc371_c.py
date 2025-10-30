import sys
import itertools

# Read all input data
data = list(map(int, sys.stdin.read().split()))
index = 0

# Read N
N = data[index]
index += 1

# Read M_G and edges for G
M_G = data[index]
index += 1
g_edges = set()
for _ in range(M_G):
    u = data[index] - 1  # Convert to 0-based
    index += 1
    v = data[index] - 1  # Convert to 0-based
    index += 1
    # Since u < v in input, after conversion still u < v
    g_edges.add((u, v))

# Read M_H and edges for H
M_H = data[index]
index += 1
h_edges = set()
for _ in range(M_H):
    a = data[index] - 1  # Convert to 0-based
    index += 1
    b = data[index] - 1  # Convert to 0-based
    index += 1
    # Since a < b in input, after conversion still a < b
    h_edges.add((a, b))

# Read the cost matrix A and store in A_cost with 0-based indices
A_cost = {}
for i in range(1, N):  # i from 1 to N-1, 1-based
    for j in range(i + 1, N + 1):  # j from i+1 to N, 1-based
        A_ij = data[index]
        index += 1
        i_zero = i - 1  # Convert to 0-based
        j_zero = j - 1  # Convert to 0-based
        A_cost[(i_zero, j_zero)] = A_ij

# Now find the minimum cost over all permutations
min_cost = float('inf')
for perm in itertools.permutations(range(N)):  # perm is mapping from g_idx to h_idx, 0-based
    # Compute inverse permutation mapping
    inv_P_map = [0] * N  # inv_P_map[h_idx] = g_idx
    for g_idx in range(N):
        h_idx = perm[g_idx]
        inv_P_map[h_idx] = g_idx
    
    # Compute cost for this permutation
    cost = 0
    for x in range(N):  # x and y are h vertices, 0-based
        for y in range(x + 1, N):  # ensure x < y
            # Get corresponding g vertices
            g_u = inv_P_map[x]
            g_v = inv_P_map[y]
            min_g = min(g_u, g_v)
            max_g = max(g_u, g_v)
            required_edge = (min_g, max_g) in g_edges  # Check if edge exists in G
            current_edge = (x, y) in h_edges  # Check if edge exists in H
            if required_edge != current_edge:
                cost += A_cost[(x, y)]  # Add toggle cost
    
    # Update minimum cost
    if cost < min_cost:
        min_cost = cost

# Output the minimum cost
print(int(min_cost))  # Cast to int to ensure integer output