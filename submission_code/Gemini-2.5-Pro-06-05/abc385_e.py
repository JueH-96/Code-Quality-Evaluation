# YOUR CODE HERE
import sys

def solve():
    """
    Solves the Snowflake Tree problem.
    """
    # It is a good practice in competitive programming to set a higher recursion limit,
    # though not strictly necessary for this iterative solution.
    sys.setrecursionlimit(3 * 10**5 + 50)

    try:
        # Read N, the number of vertices.
        line = sys.stdin.readline()
        if not line:
            return  # Handle empty input for local testing.
        N = int(line)
    except (IOError, ValueError):
        # Exit if input is malformed or unreadable.
        sys.exit()

    # The problem constraints state 3 <= N. A snowflake requires at least 3 vertices (x=1, y=1).

    # Build adjacency list and compute degrees for all vertices.
    adj = [[] for _ in range(N + 1)]
    degrees = [0] * (N + 1)

    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        degrees[u] += 1
        degrees[v] += 1

    max_snowflake_size = 0

    # Any tree with N>=3 contains a path of 3 vertices, which is a snowflake
    # with x=1, y=1, and size 3. The problem guarantees a solution exists,
    # so initializing max_snowflake_size to 0 is safe.

    # Iterate through each vertex, considering it as the potential center of the snowflake.
    for center in range(1, N + 1):
        
        # For each neighbor of 'center', it can be a branch.
        # The number of leaves it can support is its degree minus 1 (for the edge to 'center').
        # Since y must be a positive integer, the neighbor's degree must be at least 2.
        branch_leaf_counts = []
        for neighbor in adj[center]:
            leaf_count = degrees[neighbor] - 1
            if leaf_count > 0:
                branch_leaf_counts.append(leaf_count)
        
        # If no neighbor can act as a branch, this 'center' cannot form a snowflake.
        if not branch_leaf_counts:
            continue
            
        # Sort potential leaf counts in descending order to easily find the best configuration.
        branch_leaf_counts.sort(reverse=True)
        
        # We can choose k branches (x=k). To maximize size, we pick the k neighbors that
        # can support the most leaves. The number of leaves 'y' is then limited by the
        # k-th best branch.
        for i in range(len(branch_leaf_counts)):
            x = i + 1
            y = branch_leaf_counts[i]
            
            # Size of this snowflake = 1 (center) + x * (1 (branch) + y (leaves))
            current_size = 1 + x * (1 + y)
            
            if current_size > max_snowflake_size:
                max_snowflake_size = current_size
                
    # The number of vertices to delete is the total number of vertices minus the
    # size of the largest snowflake we can form.
    deleted_vertices = N - max_snowflake_size
    print(deleted_vertices)

solve()