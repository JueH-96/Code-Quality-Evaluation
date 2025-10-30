import sys
from collections import defaultdict

def solve():
    # Read the number of vertices
    N = int(sys.stdin.readline())

    # Adjacency list and degrees (using 1-based indexing for convenience)
    adj = defaultdict(list)
    degree = [0] * (N + 1)

    # Read edges and build adjacency list and degrees
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1

    # Initialize max_snowflake_size found so far.
    # The smallest possible snowflake has x=1, y=1, size = 1 + 1 + 1*1 = 3.
    # Since N >= 3 and the graph is a tree, a path of length 2 (3 vertices) exists.
    # This path is a snowflake of size 3. So, a snowflake of size at least 3 is always possible.
    max_snowflake_size = 0

    # Iterate through each vertex as a potential center of the snowflake
    for c in range(1, N + 1):
        # k is the number of neighbors of c in the original tree, representing potential branches
        k = degree[c]

        if k == 0:
            # Isolated vertex cannot be a center with x >= 1. Skip.
            continue

        # For each neighbor 'v' of potential center 'c', calculate the number of its
        # neighbors in the original tree *other than c*. These are potential leaves
        # if 'v' is chosen as a branch connected to 'c'.
        potential_leaves_per_neighbor = []
        for neighbor in adj[c]:
            # Number of neighbors of 'neighbor' excluding 'c'
            leaves_available = degree[neighbor] - 1
            potential_leaves_per_neighbor.append(leaves_available)

        # Sort the counts of potential leaves per neighbor in descending order.
        # If we choose 'x' neighbors of 'c' to be branches, to maximize 'y', we should
        # choose the 'x' neighbors that can support the most leaves.
        potential_leaves_per_neighbor.sort(reverse=True)

        # Iterate through the possible number of branches 'x' that can be formed
        # from the neighbors of 'c'. x must be at least 1.
        # The maximum possible x is the number of neighbors, k.
        for x in range(1, k + 1):
            # The x-th largest number of potential leaves among neighbors of 'c'
            # determines the maximum possible value of 'y' if we pick
            # the top 'x' neighbors as branches.
            # The sorted list is 0-indexed, so the x-th element is at index x-1.
            max_y_for_x_branches = potential_leaves_per_neighbor[x - 1]

            # A snowflake requires y >= 1. If the x-th largest potential y is < 1,
            # then any number of branches >= x will also have a max_y < 1 (since the list is sorted).
            # We can stop checking larger x values for this center 'c'.
            if max_y_for_x_branches < 1:
                break

            # Calculate the total number of vertices in this potential snowflake:
            # 1 (center) + x (branches) + x * y (leaves)
            current_snowflake_size = 1 + x + x * max_y_for_x_branches

            # Update the overall maximum snowflake size found
            max_snowflake_size = max(max_snowflake_size, current_snowflake_size)

    # The minimum number of vertices to delete is N - max_snowflake_size
    # Since N >= 3, a snowflake of size at least 3 is always possible.
    # The algorithm correctly finds a max_snowflake_size >= 3.

    print(N - max_snowflake_size)

# Execute the solve function
solve()