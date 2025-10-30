import sys

# Increase recursion depth if N was larger, but N=10 is fine.
# sys.setrecursionlimit(2000)

def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())

    # Adjacency list representation of the graph
    # adj[u] is a list of tuples (v, w) for edges from u to v with weight w
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        adj[u].append((v, w))
        adj[v].append((u, w)) # Graph is undirected

    # dist_xor[i] will store the XOR sum of the path from vertex 1 to vertex i
    # along the DFS tree. Initialize with -1 indicating not visited.
    dist_xor = [-1] * (N + 1)
    visited = [False] * (N + 1)

    # Set to store XOR sums of cycles found. Using a set automatically handles duplicates.
    cycle_generators_set = set()

    # Use a list to store the XOR sum of the path from 1 to N found by DFS.
    # Lists are mutable, allowing the DFS function to update it.
    # Initialize with [None] to indicate no path found yet.
    base_path_to_N_xor = [None]

    # DFS function to build the spanning tree, compute path XOR sums from root (1),
    # and find cycle generators (XOR sums of fundamental cycles).
    def dfs(u, current_xor, parent):
        visited[u] = True
        dist_xor[u] = current_xor

        # If we reached the destination vertex N, record the path XOR sum
        # We only need one path's XOR sum as a base. The first one found by DFS is sufficient.
        if u == N:
            if base_path_to_N_xor[0] is None:
                base_path_to_N_xor[0] = current_xor

        # Explore neighbors
        for v, w in adj[u]:
            if v == parent:
                continue # Skip the edge leading back to the parent

            if not visited[v]:
                # (u, v) is a tree edge in the DFS tree
                dfs(v, current_xor ^ w, u)
            else:
                # (u, v) is a non-tree edge (back edge or cross edge in undirected graph)
                # This edge forms a cycle with the tree paths from root to u and root to v.
                # The XOR sum of this cycle is dist_xor[u] ^ w ^ dist_xor[v].
                # Note: dist_xor[u] is equal to current_xor in this context.
                cycle_generators_set.add(current_xor ^ w ^ dist_xor[v])

    # Start DFS from vertex 1. Initial XOR sum is 0. Parent of root can be 0 or any invalid vertex index.
    dfs(1, 0, 0)

    # Get the XOR sum of the tree path from 1 to N
    # base_path_to_N_xor[0] is guaranteed to be set because the graph is connected and N >= 2.
    base_xor = base_path_to_N_xor[0]

    # Convert the set of cycle generators to a list
    cycle_generators = list(cycle_generators_set)

    # Build a basis for the cycle space using Gaussian elimination.
    # Weights are < 2^60, so the maximum possible value is 2^60 - 1.
    # The highest possible set bit is at index 59 (since 2^59 is the largest power of 2 less than 2^60).
    # We need a basis array to store vectors with MSB from bit 0 up to bit 59. Size 60 is required.
    basis = [0] * 60 # basis[i] will store a basis vector whose MSB is at bit i, or 0 if none.

    for c in cycle_generators:
        # Reduce the current cycle generator 'c' using the existing basis vectors
        # Iterate from MSB down to LSB
        for i in range(59, -1, -1):
            # Check if the i-th bit is set in c
            if (c >> i) & 1:
                # If there is no basis vector whose MSB is at bit i
                if basis[i] == 0:
                    basis[i] = c # Add c as a new basis vector with MSB at bit i
                    break # c has been successfully added/processed for this bit level
                else:
                    # There is a basis vector with MSB at bit i (basis[i])
                    # XOR c with basis[i] to eliminate the i-th bit in c.
                    # This step ensures that c becomes linearly independent of basis[i] w.r.t bit i.
                    c ^= basis[i]
        # If c is non-zero after the inner loop, it means it introduced a new independent component
        # at some bit position `i` where basis[i] was 0, and it was assigned there.
        # If c becomes 0, it was in the span of existing basis vectors.

    # Now, minimize the base_xor using the obtained basis.
    # The set of all possible XOR sums of paths from 1 to N is {base_xor ^ v | v is in the span of the basis}.
    # We want to find the minimum value in this set.
    # This minimum is found by greedily trying to turn off the most significant bits of base_xor
    # using the basis vectors, starting from the highest bit position.
    result = base_xor
    # Iterate from MSB down to LSB
    for i in range(59, -1, -1):
        # If basis[i] is non-zero, it's a basis vector whose MSB is at bit i.
        # Consider XORing the current 'result' with basis[i].
        # If this operation reduces 'result', we perform it. Otherwise, we don't.
        # The min() function implicitly checks if result ^ basis[i] < result.
        result = min(result, result ^ basis[i]) # If basis[i] is 0, result ^ 0 = result, min is result.

    # Print the minimum XOR sum
    print(result)

# Run the solve function
solve()