# YOUR CODE HERE
import sys

# Function to read input and solve the problem
def solve():
    # Read the number of vertices N
    N = int(sys.stdin.readline())
    
    # Initialize adjacency list for the graph T. 
    # Using N+1 size for 1-based indexing of vertices.
    adj = [[] for _ in range(N + 1)]
    
    # Read N-1 edges and build the adjacency list representation of the tree T
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # Compute the degree of each vertex in T
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        # The degree of vertex i is the number of neighbors it has.
        degrees[i] = len(adj[i])

    # Count the number of leaves in T. Leaves are vertices with degree 1.
    # These vertices must correspond to initial leaves that were not chosen for connection operations.
    num_leaves = 0
    for i in range(1, N + 1):
        if degrees[i] == 1:
            num_leaves += 1

    # Calculate the total number of initial stars M using the derived formula:
    # M = (N + 2 - num_leaves) / 3
    # The problem statement implies that (N + 2 - num_leaves) is always divisible by 3.
    # Use integer division //.
    M = (N + 2 - num_leaves) // 3
    
    # Initialize a list to store the levels of the initial stars
    levels = []
    
    # Identify vertices that were definitely centers of initial stars.
    # Any vertex with degree >= 3 in T must have been a center initially.
    # Its degree in T is equal to its initial degree, which is the level of the star it centered.
    num_centers_ge3 = 0
    for i in range(1, N + 1):
        if degrees[i] >= 3:
            # Add its degree (level) to the list
            levels.append(degrees[i])
            # Count how many such centers we found
            num_centers_ge3 += 1

    # The total number of stars is M. We have identified num_centers_ge3 stars with levels >= 3.
    # The remaining stars must have centers whose degrees are 2 in T.
    # These must be level-2 stars, as the minimum level is 2.
    # Calculate the count of level-2 stars.
    num_level2_stars = M - num_centers_ge3
    
    # Add level 2 to the list for each identified level-2 star.
    # Using list.extend with list multiplication for conciseness.
    levels.extend([2] * num_level2_stars)

    # The problem requires the levels to be sorted in ascending order.
    levels.sort()
    
    # Print the sorted levels separated by spaces.
    # The * operator unpacks the list elements as separate arguments to print.
    print(*(levels))

# Execute the solve function to run the program based on standard input/output.
solve()