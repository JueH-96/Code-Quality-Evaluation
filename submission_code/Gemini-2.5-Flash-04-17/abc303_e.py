import sys

# Use sys.stdin.readline for faster input
input = sys.stdin.readline

def solve():
    N = int(input())

    # Degree calculation
    # Using 1-based indexing for vertices 1..N
    degree = [0] * (N + 1)

    # Edges are N-1 lines
    for _ in range(N - 1):
        u, v = map(int, input().split())
        # No need to store adjacency list, just update degrees
        degree[u] += 1
        degree[v] += 1

    degree_one_vertices = 0
    levels = [] # This will store levels of stars whose centers have degree > 2

    # Iterate through vertices 1 to N
    for i in range(1, N + 1):
        if degree[i] == 1:
            degree_one_vertices += 1
        elif degree[i] > 2:
            # Vertices with degree > 2 are centers of stars with level = degree in T
            # A level-k star has k leaves, its center has degree k.
            # The center's degree does not change during the operation as only leaves are involved.
            # Thus, a center of a level-k star has degree k in the final tree T.
            # If k > 2, the degree in T is > 2.
            levels.append(degree[i])

    # Calculate total number of stars M.
    # Let M be the number of initial stars. Total vertices N = sum(L_i + 1) = sum(L_i) + M.
    # Number of initial edges = sum(L_i).
    # Number of edges in final tree T = N-1.
    # Number of added edges = (N-1) - sum(L_i) = (N-1) - (N-M) = M-1.
    # Each added edge connects 2 leaves. So 2 * (M-1) initial leaves are used in operations.
    # Total initial leaves = sum(L_i) = N-M.
    # Leaves NOT used in operations = (N-M) - 2 * (M-1) = N - M - 2M + 2 = N - 3M + 2.
    # These leaves NOT used in operations are exactly the vertices with degree 1 in T.
    # So, degree_one_vertices = N - 3M + 2.
    # Rearranging gives 3M = N + 2 - degree_one_vertices.
    # M = (N + 2 - degree_one_vertices) / 3.
    # Since the input is guaranteed to be valid, N + 2 - degree_one_vertices must be divisible by 3.
    M = (N + 2 - degree_one_one_vertices) // 3

    # The levels list currently contains levels >= 3 (from centers with degree > 2 in T).
    # These correspond to the stars whose centers have degree > 2 in T.
    num_centers_gt2 = len(levels)

    # The remaining stars must be level-2 stars whose centers have degree 2 in T.
    # The total number of stars is M. We already accounted for num_centers_gt2 stars.
    # The number of remaining stars is M - num_centers_gt2.
    # Since stars must have level k >= 2, these remaining stars must have level 2.
    num_centers_eq2 = M - num_centers_gt2

    # Add level 2 for each center that ends up with degree 2 in T.
    levels.extend([2] * num_centers_eq2)

    # Sort the levels in ascending order as required by the output format.
    levels.sort()

    # Print the sorted levels separated by spaces.
    print(*levels)

solve()