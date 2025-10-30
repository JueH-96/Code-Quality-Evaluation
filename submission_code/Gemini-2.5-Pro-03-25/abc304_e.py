# YOUR CODE HERE
import sys

# Set higher recursion depth limit for safety, although iterative find avoids this need.
# sys.setrecursionlimit(200010) # Commented out as iterative find is used

# Iterative find function to avoid recursion depth issues
def find(i, parent):
    """Find the representative of the set containing element i with path compression."""
    root = i
    # Find the root of the tree containing i
    while parent[root] != root:
        # Optional: Simple path compression during find phase (halving path length)
        # parent[root] = parent[parent[root]] 
        root = parent[root]
    
    # Path compression phase: Make all nodes on the path point directly to the root
    curr = i
    while parent[curr] != root:
        next_node = parent[curr] # Store the next node in the path
        parent[curr] = root      # Point current node directly to root
        curr = next_node         # Move to the next node
    return root

def unite(i, j, parent, size):
    """Unite the sets containing elements i and j using union by size heuristic."""
    # Find the representatives (roots) of the sets containing i and j
    root_i = find(i, parent)
    root_j = find(j, parent)
    
    # If i and j are already in the same set, do nothing
    if root_i != root_j:
        # Union by size: Attach the smaller tree to the root of the larger tree
        if size[root_i] < size[root_j]:
            # Swap roots to ensure root_i corresponds to the larger tree/set
            root_i, root_j = root_j, root_i
        
        # Attach root_j to root_i
        parent[root_j] = root_i
        # Update the size of the new combined set (rooted at root_i)
        size[root_i] += size[root_j]
        # The size of root_j is no longer relevant as it's not a root anymore
        return True # Return True indicating a successful merge occurred
    return False # Return False if i and j were already in the same set

def solve():
    # Use sys.stdin.readline for potentially faster input reading compared to input()
    input = sys.stdin.readline

    # Read N (number of vertices) and M (number of edges)
    N, M = map(int, input().split())

    # Initialize Disjoint Set Union (DSU) structure
    # parent[i] stores the parent of node i in the DSU forest
    # size[i] stores the size of the tree/set rooted at i (only meaningful if i is a root)
    # Use 1-based indexing for vertices 1 to N
    parent = list(range(N + 1))
    size = [1] * (N + 1) # Initially, each vertex is in its own set of size 1

    # Process the initial M edges to build the connected components of graph G
    for _ in range(M):
        u, v = map(int, input().split())
        # Unite the sets containing u and v. This reflects adding an undirected edge (u, v).
        unite(u, v, parent, size)

    # Read K, the number of pairs (x_i, y_i) that must remain disconnected
    K = int(input())
    
    # Store the pairs of component representatives that correspond to the forbidden connections
    # Use a set for O(1) average time complexity lookups
    forbidden_pairs = set()
    for _ in range(K):
        x, y = map(int, input().split())
        # Find the representatives (roots) of the components containing x and y
        root_x = find(x, parent)
        root_y = find(y, parent)
        
        # The problem statement guarantees that the initial graph G is "good",
        # meaning for all i, x_i and y_i are not connected. Thus, root_x != root_y.
        # We store the pair of representatives canonically (sorted) to ensure that
        # the order doesn't matter, representing the unordered pair {root_x, root_y}.
        if root_x != root_y: # This check is technically redundant due to the guarantee
             if root_x > root_y:
                # Ensure root_x is always the smaller index
                root_x, root_y = root_y, root_x
             # Add the canonical pair to the set of forbidden pairs of components
             forbidden_pairs.add((root_x, root_y))
        # No else case needed because problem guarantees G is good. 
        # If root_x == root_y, it would contradict the input specification.
        
    # Read Q, the number of queries
    Q = int(input())
    
    # Prepare a list to store the answers ("Yes" or "No") for each query
    results = []
    for _ in range(Q):
        # Read the query edge (p, q) for the i-th query
        p, q = map(int, input().split())
        
        # Find the representatives of the components containing p and q in the graph G
        root_p = find(p, parent)
        root_q = find(q, parent)
        
        # Adding the edge (p, q) connects the components containing p and q.
        # This potentially creates a path between some forbidden pair (x_j, y_j).
        # A path is created iff p is connected to x_j and q is connected to y_j (in G),
        # OR p is connected to y_j and q is connected to x_j (in G).
        # This is equivalent to checking if the pair of components {C(p), C(q)}
        # is equal to one of the forbidden pairs of components {C(x_j), C(y_j)}.
        
        # Canonicalize the pair (root_p, root_q) by sorting the representatives
        if root_p > root_q:
            root_p, root_q = root_q, root_p
            
        # Check if this canonical pair of component representatives exists in the forbidden set.
        # Important note: If p and q are already in the same component (root_p == root_q),
        # the canonical pair becomes (root_p, root_p). Since all pairs in `forbidden_pairs`
        # are of the form (a, b) with a < b (because G is good and x_j != y_j),
        # the pair (root_p, root_p) will never be found in `forbidden_pairs`.
        # This correctly handles the case where adding edge (p, q) doesn't connect
        # any new pair of components, and thus the graph remains good.
        if (root_p, root_q) in forbidden_pairs:
            # If the pair {Rep(p), Rep(q)} matches a forbidden pair {Rep(x_j), Rep(y_j)},
            # adding the edge (p, q) makes the graph G^{(i)} not good.
            results.append("No")
        else:
            # Otherwise, adding the edge (p, q) does not violate any forbidden connection condition.
            # The graph G^{(i)} remains good.
            results.append("Yes")

    # Print all results, each on a new line.
    # Using sys.stdout.write combined with '
'.join for potentially faster output.
    sys.stdout.write('
'.join(results) + '
')

# Call the main solve function to run the program
solve()