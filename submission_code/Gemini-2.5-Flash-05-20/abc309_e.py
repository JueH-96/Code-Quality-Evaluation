import sys

# Increase the recursion limit for deep trees.
# The maximum depth of the tree can be N-1. N is up to 3 * 10^5.
# A buffer is added to ensure enough stack space.
sys.setrecursionlimit(3 * 10**5 + 100)

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # Build the adjacency list for children (parent -> children).
    # children[u] will store a list of direct children of person u.
    children = [[] for _ in range(N + 1)]
    if N > 1: # Parent information p_i is provided for i from 2 to N.
        # The input line for parents contains p_2, p_3, ..., p_N.
        # When read as a list, parents_list[0] corresponds to p_2, parents_list[1] to p_3, etc.
        # So, parents_list[i-2] gives the parent of person i.
        parents_list = list(map(int, sys.stdin.readline().split()))
        for i in range(2, N + 1):
            parent_of_i = parents_list[i - 2]
            children[parent_of_i].append(i)

    # Store insurance policies by the person who bought them.
    # insurance_policies_by_buyer[x] will be a list of y_i values
    # for all policies bought by person x.
    insurance_policies_by_buyer = [[] for _ in range(N + 1)]
    for _ in range(M):
        x, y = map(int, sys.stdin.readline().split())
        insurance_policies_by_buyer[x].append(y)

    # Calculate depths of all nodes from person 1 (the root).
    # depth[u] will store the depth of node u, with depth[1] = 0.
    depth = [-1] * (N + 1)
    
    # Using BFS to calculate depths as it's typically robust against stack overflow
    # for very deep trees, although with the increased recursion limit, DFS would also be fine.
    # The queue stores tuples of (node, current_depth).
    q = [(1, 0)] 
    depth[1] = 0
    head = 0 # An index acting as a pointer for an efficient in-place queue (list as queue)
    while head < len(q):
        u, d = q[head]
        head += 1
        for v in children[u]:
            if depth[v] == -1: # Only visit unvisited nodes
                depth[v] = d + 1
                q.append((v, d + 1))
    
    # Initialize a counter for the total number of people covered.
    covered_count = 0
    
    # Define the main DFS function to traverse the tree and determine coverage.
    # u: The current person (node) being visited.
    # max_inherited_reach_depth: This value represents the maximum 'depth + generations'
    #                           value achieved by any insurance policy bought by an ancestor
    #                           of 'u' (not including 'u' itself yet).
    #                           This is the deepest level a node can be at
    #                           and still be covered due to an ancestor's policy.
    def dfs(u, max_inherited_reach_depth):
        nonlocal covered_count

        # Calculate the maximum reach depth considering insurances bought by the current node 'u'
        # and all its ancestors.
        current_node_max_reach_depth = max_inherited_reach_depth
        for y_val in insurance_policies_by_buyer[u]:
            # An insurance policy bought by 'u' covers people up to depth[u] + y_val.
            # We take the maximum of this with the inherited reach.
            current_node_max_reach_depth = max(current_node_max_reach_depth, depth[u] + y_val)
        
        # Check if the current node 'u' itself is covered.
        # Person 'u' is covered if their own depth is less than or equal to the maximum allowed depth
        # originating from 'u' or any of its ancestors.
        if depth[u] <= current_node_max_reach_depth:
            covered_count += 1
        
        # Recursively call DFS for each child of 'u'.
        # The 'current_node_max_reach_depth' is passed down,
        # so children inherit the consolidated maximum reach information.
        for v in children[u]:
            dfs(v, current_node_max_reach_depth)

    # Start the DFS from the root (person 1).
    # The initial 'max_inherited_reach_depth' is set to -1.
    # This value ensures that person 1 is counted as covered ONLY if they themselves buy an insurance policy.
    # (Since depth[1] is 0, '0 <= -1' is false, so person 1 needs a policy with y_val >= 0 to be covered).
    dfs(1, -1)

    # Print the total count of covered people to standard output.
    sys.stdout.write(str(covered_count) + "
")

# Call the solve function to execute the program.
solve()