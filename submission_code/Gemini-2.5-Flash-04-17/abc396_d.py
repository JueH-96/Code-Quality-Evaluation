import sys

# Increase recursion depth for DFS if needed. For N=10, the default limit should be fine,
# but it's good practice for potential larger N in similar problems or different graph structures.
# sys.setrecursionlimit(2000)

def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())
    
    # Use 0-indexed vertices internally (0 to N-1)
    # Vertex 1 is index 0, Vertex N is index N-1
    
    # Adjacency list for the graph
    # graph[u] will store a list of tuples (v, w) for edges connecting u to v with weight w
    graph = [[] for _ in range(N)]
    
    # Read edges and build the graph adjacency list
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        u -= 1 # Convert to 0-indexed
        v -= 1 # Convert to 0-indexed
        graph[u].append((v, w))
        graph[v].append((u, w))

    # Step 1 & 2: DFS to build a spanning tree, compute distances from vertex 0, and find cycle values
    
    # dist[i] stores the XOR sum of edge weights on the unique path from vertex 0 to vertex i
    # in the spanning tree discovered by DFS.
    # Initialize with -1 to indicate unvisited vertices.
    dist = [-1] * N 
    
    # List to store the XOR sums of fundamental cycles found by DFS back edges.
    cycle_values = []
    
    # visited status is implicitly checked by dist[v] != -1.
    # The parent argument in DFS prevents treating the edge leading back to the immediate parent as a cycle edge.

    def dfs(u, current_xor, parent):
        """
        Performs DFS starting from vertex u.
        current_xor: XOR sum of edge weights from the starting vertex (0) to u in the tree path.
        parent: The vertex from which u was visited in the DFS tree.
        """
        dist[u] = current_xor
        
        for v, w in graph[u]:
            if dist[v] == -1: # If v is not visited yet
                # (u, v) is a tree edge in this DFS traversal. Recurse on v.
                dfs(v, current_xor ^ w, u)
            # If v is visited AND v is not the parent from which we arrived at u
            elif v != parent: 
                # (u, v) is a non-tree edge (a back edge) relative to the current DFS tree.
                # This back edge completes a cycle. The cycle includes the edge (u, v)
                # and the path in the spanning tree between u and v.
                # The XOR sum of the path in the tree from root (0) to u is dist[u].
                # The XOR sum of the path in the tree from root (0) to v is dist[v].
                # The XOR sum of the path in the tree between u and v is dist[u] ^ dist[v].
                # The cycle XOR sum is (path_xor(u to v in tree)) ^ edge_w(v to u)
                # Cycle XOR = (dist[u] ^ dist[v]) ^ w
                # Note: current_xor is equal to dist[u]
                cycle_values.append(current_xor ^ dist[v] ^ w)

    # Start DFS from vertex 0 (original vertex 1)
    # The parent of the root (0) is considered -1 as it has no parent.
    dfs(0, 0, -1) 

    # The XOR sum of the path from original vertex 1 (index 0) to original vertex N (index N-1)
    # in the spanning tree found by DFS is stored in dist[N-1].
    min_xor_sum = dist[N-1]

    # Step 3 & 4: Build a linear basis for the cycle space and prepare for minimization

    # The basis will store a set of linearly independent vectors whose XOR span
    # covers all possible XOR sums of cycles in the graph.
    basis = []

    def insert_basis(val, basis):
        """
        Attempts to insert a value into the linear basis.
        Reduces the value using existing basis vectors. If the reduced value is non-zero,
        it's added to the basis and the basis is kept sorted.
        """
        # Try to reduce val using existing basis vectors.
        # The property min(a, a ^ b) effectively performs Gaussian elimination:
        # if XORing with b reduces a (i.e., a ^ b < a), update a to a ^ b.
        # This greedily removes the highest set bit if a basis vector 'b' can do so.
        for b in basis:
            val = min(val, val ^ b) 

        # If val is non-zero after reduction, it is linearly independent of the current basis.
        if val > 0:
            # Add the reduced, non-zero value to the basis.
            basis.append(val)
            # Keep the basis sorted in descending order. This is crucial for the
            # greedy minimization step later. Sorting helps process bits from MSB downwards
            # implicitly, as higher values tend to have higher MSBs.
            basis.sort(reverse=True) 

    # Insert all collected cycle values into the basis
    for cycle_val in cycle_values:
        insert_basis(cycle_val, basis)

    # Step 5: Minimize the base path XOR sum using the cycle basis

    # The current min_xor_sum holds the XOR sum of one specific path from 0 to N-1 (the tree path).
    # Any other path from 0 to N-1 has an XOR sum equal to (this tree path XOR sum) XORed
    # with some element from the span of the cycle basis.
    # We want to find the minimum value achievable by XORing the initial path sum
    # with any element from the cycle space span. This minimum value is found by
    # greedily reducing the initial path sum using the basis vectors.
    
    # Iterate through the basis vectors. Since the basis is sorted descending,
    # processing them in this order ensures the greedy choice is optimal.
    for b in basis:
        # If XORing the current minimum value with a basis vector results
        # in a smaller value, update the minimum value.
        # This greedy step tries to make the highest possible bit 0.
        # If current_min_xor_sum has a 1 at the MSB position of b, and b also has a 1 there,
        # XORing with b flips that bit to 0, reducing the value.
        # If current_min_xor_sum has a 0 at the MSB position of b, XORing with b flips
        # that bit to 1, increasing the value. The min() function chooses not to do this.
        min_xor_sum = min(min_xor_sum, min_xor_sum ^ b)

    # Output the final minimum XOR sum found
    print(min_xor_sum)

# Execute the solve function
solve()