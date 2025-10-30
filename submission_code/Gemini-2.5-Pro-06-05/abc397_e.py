import sys
from collections import defaultdict

# An iterative approach is used to avoid Python's recursion limit
# for deep trees, which can occur given the problem constraints.

def solve():
    """
    Solves the Tree Path Decomposition problem.
    """
    line = sys.stdin.readline()
    if not line:
        return
    N, K = map(int, line.split())

    V = N * K

    # If K=1, any tree with N vertices can be decomposed into N paths of length 1.
    if K == 1:
        # We must still consume the rest of the input lines.
        for _ in range(V - 1):
            sys.stdin.readline()
        print("Yes")
        return

    # A single-vertex tree is trivially decomposable.
    if V == 1:
        print("Yes")
        return

    # Build adjacency list for the tree.
    adj = defaultdict(list)
    for _ in range(V - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # Root the tree at vertex 1 and establish parent-child relationships using BFS.
    parent = [-1] * (V + 1)
    q = [1]
    visited_bfs = {1}
    parent[1] = 0  # Sentinel parent for the root
    head = 0
    traversal_order = []
    while head < len(q):
        u = q[head]
        head += 1
        traversal_order.append(u)
        for v in adj[u]:
            if v not in visited_bfs:
                visited_bfs.add(v)
                parent[v] = u
                q.append(v)

    # Calculate subtree sizes.
    # Iterating in reverse BFS order ensures that when we calculate size[u],
    # the sizes of all its children have already been computed.
    size = [-1] * (V + 1)
    for u in reversed(traversal_order):
        s = 1
        for v in adj[u]:
            if parent[v] == u:  # v is a child of u
                s += size[v]
        size[u] = s

    # Construct a new graph with only the "kept" edges.
    # An edge (u, v) (u=parent) is kept if size[v] is not a multiple of K.
    adj_kept = defaultdict(list)
    for v in range(1, V + 1):
        if parent[v] != 0:  # If v is not the root
            if size[v] % K != 0:
                u = parent[v]
                adj_kept[u].append(v)
                adj_kept[v].append(u)
    
    # Verify that the components of the new graph are all paths of length K.
    visited_comp = [False] * (V + 1)
    for i in range(1, V + 1):
        if not visited_comp[i]:
            component_nodes = []
            q_comp = [i]
            visited_comp[i] = True
            head_comp = 0
            
            # Find all nodes in the current connected component using BFS.
            while head_comp < len(q_comp):
                u = q_comp[head_comp]
                head_comp += 1
                component_nodes.append(u)
                for v in adj_kept[u]:
                    if not visited_comp[v]:
                        visited_comp[v] = True
                        q_comp.append(v)

            # Check 1: The component must have exactly K vertices.
            if len(component_nodes) != K:
                print("No")
                return
            
            # Check 2: The component must be a path (max degree <= 2).
            for u in component_nodes:
                if len(adj_kept[u]) > 2:
                    print("No")
                    return

    # If all components are valid paths of length K, the decomposition is possible.
    print("Yes")

solve()