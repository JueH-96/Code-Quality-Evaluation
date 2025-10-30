import sys

# Increase recursion depth for potentially deep trees (path graphs)
sys.setrecursionlimit(200005) 

def compute_subtree_size(u, parent, adj, subtree_size):
    """Helper function to compute subtree size for each node."""
    size = 1
    for v in adj[u]:
        if v != parent:
            size += compute_subtree_size(v, u, adj, subtree_size)
    subtree_size[u] = size
    return size

def solve():
    N, K = map(int, sys.stdin.readline().split())
    NK = N * K

    # Special case: If K=1, any set of NK vertices can be decomposed into NK paths of length 1.
    # We need N paths. This is only possible if NK = N, which implies K=1.
    # If K=1, we need N paths of length 1. We have N vertices. This is always possible.
    # (Vertices are 1 to N. Each vertex forms a path of length 1).
    # The condition about edges P_{i,j} and P_{i,j+1} is vacuously true for length 1 paths (j goes from 1 to K-1=0).
    if K == 1:
        # Read edges to consume input, even though graph structure doesn't matter.
        for _ in range(NK - 1):
            sys.stdin.readline()
        print("Yes")
        return

    # For K > 1, graph structure matters. Read edges and build adjacency list.
    adj = [[] for _ in range(NK + 1)]
    for _ in range(NK - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # Precompute subtree sizes. Root arbitrarily at 1.
    subtree_size = [0] * (NK + 1)
    compute_subtree_size(1, 0, adj, subtree_size)

    # DFS function for decomposition check
    def dfs(u, parent):
        """
        Checks if the subtree rooted at u can be decomposed into full paths of length K
        plus potentially one partial path of length `subtree_size[u] % K` ending at u
        and requiring connection upwards via the parent edge.

        Returns:
            The value `subtree_size[u] % K` if the decomposition is possible for the subtree,
            -1 otherwise.
        """
        my_size = subtree_size[u]
        required_rem = my_size % K

        # Collect the actual remainders returned by children subtrees
        actual_child_rems = []
        # Count how many children subtrees require their parent edge (u,v) to be a path edge.
        # This happens if dfs(v, u) returns a value > 0.
        path_edges_from_children = 0

        for v in adj[u]:
            if v != parent:
                child_rem = dfs(v, u)
                if child_rem == -1:
                    # If any child subtree cannot be decomposed, this subtree also cannot.
                    return -1

                actual_child_rems.append(child_rem)
                if child_rem > 0:
                    # If a child v returns a remainder > 0, it means the subtree rooted at v
                    # forms a partial path of that length ending at v. The edge (u, v)
                    # must be a path edge to extend this segment to u.
                    path_edges_from_children += 1

        # Now analyze node u based on the results from children and its required remainder.
        # `required_rem` is the length of the partial path segment that must end at u
        # and be connected upwards, based on the total vertex count in the subtree.

        if required_rem > 0:
            # Node u must be the endpoint of a path segment of length `required_rem` ending at u.
            # This path segment uses `required_rem` vertices from the subtree.
            # This segment needs one path edge upwards (to the parent) unless u is the root.
            # It also needs path edges downwards to connect to the rest of its segment.
            # If required_rem == 1: the segment is just (u). It needs 0 path edges from children
            #                      to form this segment (the segment is just u itself).
            # If required_rem > 1: the segment is (... -> v -> u). It needs exactly one path edge from a child v,
            #                      where v is the (required_rem - 1)-th node in the segment.
            required_from_children = 0 if required_rem == 1 else 1

            if path_edges_from_children != required_from_children:
                return -1 # Mismatch in the number of path edges from children required for the upward segment.

            # All checks passed for the required_rem > 0 case.
            return required_rem

        else: # required_rem == 0
            # The subtree rooted at u (including u) must be perfectly decomposable into full paths of length K.
            # Node u is NOT the endpoint of a partial path needing upward connection.
            # U must be an internal node of a path (path degree 2) or an endpoint of a path
            # contained entirely within the subtree (path degree 1).
            # These path edges must come entirely from children subtrees.
            # Number of path edges from children must be 1 (endpoint) or 2 (internal node).
            if path_edges_from_children != 1 and path_edges_from_children != 2:
                return -1 # Mismatch in the number of path edges from children for internal decomposition.

            # All checks passed for the required_rem == 0 case.
            return 0

    # The entire tree must be perfectly decomposable into N paths of length K.
    # The subtree rooted at the root (vertex 1) is the entire tree, with size NK.
    # The required remainder for the root is NK % K, which is always 0.
    # The DFS call on the root must return 0 for the entire tree to be decomposable.
    result = dfs(1, 0)

    if result == 0:
        print("Yes")
    else:
        print("No")

# Read input and call the solve function
solve()