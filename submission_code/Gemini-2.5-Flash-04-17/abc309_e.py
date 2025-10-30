# YOUR CODE HERE
import sys
# Increase recursion depth for potentially deep trees
# Maximum depth can be N (3e5), need sufficient limit.
sys.setrecursionlimit(400000)

def solve():
    # Read N (number of people) and M (number of insurance policies)
    # N: 2 <= N <= 3 * 10^5
    # M: 1 <= M <= 3 * 10^5
    N, M = map(int, sys.stdin.readline().split())

    # Build adjacency list representation of the family tree
    # adj[u] will store a list of children of person u.
    # People are 1-indexed (1 to N), so we use lists of size N + 1.
    adj = [[] for _ in range(N + 1)]
    if N > 1: # Persons 2 to N have parents. Person 1 is the root implicitly.
        # Read the parents p_2, p_3, ..., p_N.
        # p_i is the parent of person i. The input is a list p_2, ..., p_N.
        p = list(map(int, sys.stdin.readline().split()))
        # The list p has N-1 elements. p[0] is parent of person 2, ..., p[N-2] is parent of person N.
        for i in range(2, N + 1):
            parent = p[i - 2] # Person i's parent p_i is at index i-2 in the input list.
            adj[parent].append(i)

    # Store insurance policies bought by each person.
    # insurances_bought[u] will store a list of y-values for insurances bought by person u.
    # People are 1-indexed (1 to N), so we use lists of size N + 1.
    # y_i: 1 <= y_i <= 3 * 10^5
    insurances_bought = [[] for _ in range(N + 1)]
    for _ in range(M):
        x, y = map(int, sys.stdin.readline().split())
        insurances_bought[x].append(y)

    # Boolean array to track which people are covered by at least one insurance.
    # covered[u] is True if person u is covered, False otherwise.
    # People are 1-indexed (1 to N), so we use a list of size N + 1.
    covered = [False] * (N + 1)

    # Depth First Search function to determine coverage recursively.
    # u: The current person being visited (1-indexed).
    # current_depth: The depth of person u in the tree. The root (person 1) is at depth 0.
    # max_reach_from_ancestors: This value represents the maximum depth `v` can have such that `v`
    #                          is covered by an insurance bought by any *proper* ancestor `a` of `u`.
    #                          Specifically, it is the maximum value of (depth[a] + y_a) among all
    #                          proper ancestors 'a' of u that bought insurance (a, y_a).
    #                          We initialize this with -1, a value strictly less than any possible depth (min depth is 0).
    def dfs(u, current_depth, max_reach_from_ancestors):
        # Calculate the maximum depth reachable by insurances bought *by the current person u*.
        # If person u bought insurances (u, y), they cover descendants down to depth `current_depth + y`.
        # If u bought multiple insurances, the one with the maximum y determines the reach from u.
        my_max_reach_depth = -1 # Initialize such that it doesn't affect the max if u bought no insurance.
        if insurances_bought[u]:
             # The maximum depth covered by an insurance bought by u is current_depth + max_y.
             # Since y_i >= 1, this value will be >= current_depth + 1.
             my_max_reach_depth = current_depth + max(insurances_bought[u])

        # The overall maximum depth reached considering insurances from all ancestors *including* u.
        # This is the maximum of (depth[x] + y_x) for any ancestor 'x' of u (including u)
        # that bought an insurance (x, y_x).
        # This value is passed down to the children of u.
        current_max_reach_depth = max(max_reach_from_ancestors, my_max_reach_depth)

        # A person u is covered if their own depth is less than or equal to the maximum depth
        # covered by any insurance policy issued by any of their ancestors (including themselves)
        # on the path from the root.
        if current_depth <= current_max_reach_depth:
            covered[u] = True

        # Recurse for each child of u.
        # For a child v of u, u is an ancestor. The maximum reach for v derived from its
        # ancestors (excluding v itself) is exactly the `current_max_reach_depth` calculated for u.
        for v in adj[u]:
            dfs(v, current_depth + 1, current_max_reach_depth)

    # Start the DFS traversal from the root of the tree (person 1).
    # The root is at depth 0. Initially, there are no proper ancestors, so the initial
    # max_reach_from_ancestors is -1 (or any value strictly less than 0).
    dfs(1, 0, -1)

    # After the DFS completes, the `covered` array indicates which people are covered.
    # Count the number of people marked as covered (from index 1 to N).
    count = sum(covered[1:])

    # Print the final count.
    print(count)

# Execute the main solve function.
solve()