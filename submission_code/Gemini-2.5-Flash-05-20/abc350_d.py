# YOUR CODE HERE
import sys

# It's good practice to increase the recursion limit for DSU with path compression
# especially for large N, as default limits (e.g., 1000 or 3000) can be too low.
# A limit of N + a small buffer (e.g., 100) is usually safe for N up to 2*10^5.
sys.setrecursionlimit(2 * 10**5 + 100) 

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # parent[i] stores the parent of element i. If parent[i] == i, i is a root.
    parent = list(range(N + 1))
    # sz[i] stores the size of the set if i is the root of the set.
    # Initialized to 1, as each node starts in its own set.
    sz = [1] * (N + 1)
    # edges[i] stores the number of initial edges within the component if i is the root.
    # Initialized to 0, as a single node component has no edges.
    edges = [0] * (N + 1)

    def find(i):
        # Base case: if i is the root of its set
        if parent[i] == i:
            return i
        # Path compression: make the parent of i the root of its set
        # This flattens the tree, making future lookups faster.
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)

        # If i and j are already in the same set (same root)
        if root_i == root_j:
            # The edge (i, j) is being added within an already connected component.
            # We still need to count this edge for the total initial edges in that component.
            edges[root_i] += 1 
            return False # No merge of components occurred
        else:
            # Union by size: attach the smaller tree under the root of the larger tree.
            # This strategy helps keep the tree relatively flat, improving performance.
            if sz[root_i] < sz[root_j]:
                root_i, root_j = root_j, root_i

            parent[root_j] = root_i # Make root_i the new parent of root_j (merging root_j's component into root_i's)
            sz[root_i] += sz[root_j] # Update the size of the new combined component
            # Sum the edges from both merged components, and add the new edge (i, j) 
            # that connected them to form the larger component.
            edges[root_i] += edges[root_j] + 1 
            return True # A merge of two distinct components occurred

    # Process all initial friendships provided in the input
    for _ in range(M):
        A, B = map(int, sys.stdin.readline().split())
        union(A, B)

    total_new_friendships = 0
    # Iterate through all users to find the roots of all distinct connected components.
    # A node `i` is a root if `parent[i] == i`.
    for i in range(1, N + 1):
        if parent[i] == i: # If i is the root of a component
            k = sz[i]       # k is the number of users (nodes) in this component
            E_k = edges[i]  # E_k is the number of initial friendships (edges) within this component
            
            # In a fully connected component (a clique) with k nodes, 
            # the total number of possible friendships is k * (k - 1) / 2.
            # The number of new friendships formed for this component is this maximum possible 
            # minus the friendships that already exist within it.
            # For k=1 (isolated node), k*(k-1)//2 evaluates to 0, which correctly adds 0 new friendships.
            total_new_friendships += (k * (k - 1) // 2) - E_k
            
    # Print the final calculated maximum number of new friendships
    print(total_new_friendships)

# Call the solve function to run the program
solve()