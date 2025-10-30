# YOUR CODE HERE
import sys

# For deep recursion in tree/graph problems, it's a good practice to increase the
# recursion limit. The maximum recursion depth can be N in a path graph, and
# N can be up to 3*10^5, so we set the limit higher.
sys.setrecursionlimit(3 * 10**5 + 50)

def solve():
    """
    Reads graph input, solves the problem, and prints the result.
    """
    # Read the number of vertices. Using sys.stdin.readline for efficiency.
    try:
        line = sys.stdin.readline()
        if not line: return # Handle empty input
        N = int(line)
    except (ValueError, IndexError):
        return

    # Create an adjacency list to represent the tree.
    # Vertices are 1-indexed, so we use a list of size N+1 for convenience.
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # A leaf is a vertex with degree at most 1.
    # If vertex 1 is already a leaf, it can be deleted in the first operation.
    # In a tree with N>=2, a leaf has degree 1. If N=1, degree is 0.
    # This check covers both cases.
    if len(adj[1]) <= 1:
        print(1)
        return

    def dfs_size(u, p):
        """
        Recursively calculates the size of the subtree rooted at `u`.
        The parent `p` is passed to prevent the traversal from going back up
        the tree, ensuring each edge is traversed once downwards.
        The size of a subtree is 1 (for its root `u`) plus the sum of the
        sizes of the subtrees of its children.
        """
        size = 1
        for v in adj[u]:
            if v != p:
                size += dfs_size(v, u)
        return size

    # The core logic is that to delete vertex 1, it must first become a leaf.
    # To minimize operations, we preserve the largest subtree connected to
    # vertex 1 and delete all other smaller subtrees.

    # Calculate the sizes of all subtrees attached to the neighbors of vertex 1.
    subtree_sizes = []
    for neighbor in adj[1]:
        # The parent of each neighbor in this context is vertex 1.
        size = dfs_size(neighbor, 1)
        subtree_sizes.append(size)

    # The total number of operations is N minus the size of the part we keep.
    # We keep the largest branch.
    # This is derived from:
    # ops = 1 (for vertex 1) + sum_of_sizes(branches to remove)
    #     = 1 + ( (N - 1) - max_branch_size )
    #     = N - max_branch_size
    
    max_branch_size = max(subtree_sizes)
    result = N - max_branch_size
    print(result)

# Run the solution
solve()