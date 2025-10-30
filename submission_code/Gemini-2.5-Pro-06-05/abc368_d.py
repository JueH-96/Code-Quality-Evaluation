import sys

# Set a higher recursion limit for deep trees. The default limit (e.g., 1000)
# might be too low for N up to 2*10^5 in a path graph.
sys.setrecursionlimit(2 * 10**5 + 5)

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    # Use sys.stdin.readline for faster I/O
    input = sys.stdin.readline
    
    try:
        line = input()
        if not line: return # Handle empty input at EOF
        N, K = map(int, line.split())
    except (IOError, ValueError):
        return

    # Build adjacency list representation of the tree
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # Read the K special vertices
    special_nodes = list(map(int, input().split()))

    # If there's only one special vertex, the minimal tree consists of that vertex alone.
    if K == 1:
        print(1)
        return

    # --- Precomputation for LCA using Binary Lifting ---
    
    # MAX_LOG_N determines the size of the parent table. N.bit_length() is ceil(log2(N+1)).
    MAX_LOG_N = N.bit_length()
    
    # Data structures for DFS traversal and LCA calculation
    depth = [-1] * (N + 1)
    # parent[v][j] stores the 2^j-th ancestor of vertex v
    parent = [[0] * MAX_LOG_N for _ in range(N + 1)]
    # tin[v] stores the discovery time of v in a DFS traversal
    tin = [-1] * (N + 1)
    timer = 0

    def dfs(v, p, d):
        nonlocal timer
        tin[v] = timer
        timer += 1
        
        depth[v] = d
        parent[v][0] = p # The 2^0 = 1st ancestor is the direct parent
        
        for neighbor in adj[v]:
            if neighbor != p:
                dfs(neighbor, v, d + 1)

    # We can root the tree arbitrarily, e.g., at vertex 1.
    # The parent of the root can be set to itself.
    dfs(1, 1, 0)
    
    # Fill the binary lifting table for efficient LCA queries
    for j in range(1, MAX_LOG_N):
        for i in range(1, N + 1):
            # The 2^j-th ancestor is the 2^(j-1)-th ancestor of the 2^(j-1)-th ancestor
            p_i_j_minus_1 = parent[i][j - 1]
            parent[i][j] = parent[p_i_j_minus_1][j - 1]

    def lca(u, v):
        # Bring the deeper node up to the same level
        if depth[u] < depth[v]:
            u, v = v, u
            
        # Lift u to the same depth as v
        for j in range(MAX_LOG_N - 1, -1, -1):
            if depth[u] - (1 << j) >= depth[v]:
                u = parent[u][j]
        
        # If v was an ancestor of the original u, they are now the same.
        if u == v:
            return u
            
        # Lift u and v up together until their parents are the same.
        # This parent will be the LCA.
        for j in range(MAX_LOG_N - 1, -1, -1):
            if parent[u][j] != parent[v][j]:
                u = parent[u][j]
                v = parent[v][j]
        
        return parent[u][0]

    # --- Main Calculation ---

    # Sort special vertices by their DFS discovery time. This orders them
    # according to their position on the "perimeter" of the minimal subtree.
    special_nodes.sort(key=lambda node: tin[node])
    
    # The number of edges in the minimal subtree is given by the formula:
    # NumEdges = sum(depth(U_i)) - sum(depth(LCA(U_i, U_{i-1})))
    # where U is the tin-sorted list of special nodes.
    
    sum_depth_special = 0
    sum_depth_lca = 0
    
    for i in range(K):
        u = special_nodes[i]
        # v is the previous node in the cyclic order. For i=0, special_nodes[-1]
        # correctly gives the last element of the list in Python.
        v = special_nodes[i-1] 
        
        sum_depth_special += depth[u]
        sum_depth_lca += depth[lca(u, v)]
        
    num_edges = sum_depth_special - sum_depth_lca
    
    # The number of vertices in any tree is the number of edges plus one.
    num_vertices = num_edges + 1
    
    print(num_vertices)

# Standard entry point for the script
if __name__ == "__main__":
    solve()