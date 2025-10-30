import sys

# Function to calculate the number of nodes in a subtree at a specific depth
# below a given root_node, up to N_limit.
# root_node: the starting node of the subtree
# depth: how many levels down from root_node to count
# N_limit: the maximum vertex number in the tree (N)
def count_nodes_at_depth(root_node, depth, N_limit):
    if depth < 0:
        return 0
    if root_node == 0: # Vertex 0 does not exist
        return 0

    # Calculate the range of nodes at 'depth' levels below root_node
    # The leftmost node at 'depth' levels is root_node * (2^depth)
    # The rightmost node is root_node * (2^depth) + (2^depth) - 1
    
    # Use bit shift for 2^depth for efficiency
    power_of_2 = 1 << depth
    
    start_range = root_node * power_of_2
    
    # If the starting node itself is beyond N_limit, no nodes exist in this range
    if start_range > N_limit:
        return 0
    
    end_range = start_range + power_of_2 - 1
    
    # The actual end of the range is limited by N_limit
    actual_end_node = min(end_range, N_limit)
    
    # Count the number of nodes in the valid range
    # (actual_end_node - start_range + 1)
    return max(0, actual_end_node - start_range + 1)

def solve():
    N, X, K = map(int, sys.stdin.readline().split())

    if K == 0:
        print(1) # Only X itself is at distance 0 from X
        return

    ans = 0

    # Part 1: Count descendants of X at distance K
    # These are nodes in the subtree rooted at X, K levels down.
    ans += count_nodes_at_depth(X, K, N)

    # Part 2: Iterate upwards from X, considering sibling subtrees
    # and ancestors on the path from X to the root.
    current_node = X
    depth_from_X_to_current_node = 0 # Distance from original X to current_node
    
    while True:
        # Move up one step to the parent of current_node
        depth_from_X_to_current_node += 1
        parent = current_node // 2

        # If parent is 0, we've gone above the root (vertex 1)
        if parent == 0:
            break

        # rem_dist_from_parent is the remaining distance required from 'parent'
        # to reach a node at K distance from X.
        # Total path: X -> ... -> parent (depth_from_X_to_current_node steps) -> Y (rem_dist_from_parent steps)
        rem_dist_from_parent = K - depth_from_X_to_current_node

        # If remaining distance is negative, target Y would be an ancestor of parent,
        # but we are looking for nodes downwards from parent (or parent itself).
        # Also, if we've gone too far up, no valid path of length K.
        if rem_dist_from_parent < 0:
            break

        # Case 2a: Parent itself is at distance K from X
        # This occurs if rem_dist_from_parent is 0.
        if rem_dist_from_parent == 0:
            ans += 1
        
        # Case 2b: Count nodes in the sibling's subtree
        # Sibling of 'current_node' (which is a child of 'parent').
        # Target node Y is reached by: X -> ... -> parent -> sibling -> Y
        # dist(parent, Y) = rem_dist_from_parent
        # so dist(sibling, Y) = rem_dist_from_parent - 1 (1 step from parent to sibling)
        
        dist_down_from_sibling = rem_dist_from_parent - 1
        
        # Determine the sibling node
        # If current_node is a left child (even), sibling is right child (+1)
        # If current_node is a right child (odd), sibling is left child (-1)
        # Note: current_node is parent's child
        if current_node % 2 == 0: # current_node is left child (parent*2)
            sibling = (parent * 2) + 1
        else: # current_node is right child (parent*2 + 1)
            sibling = (parent * 2)
        
        # Add nodes from the sibling's subtree
        ans += count_nodes_at_depth(sibling, dist_down_from_sibling, N)

        # Move up to the parent for the next iteration
        current_node = parent

    print(ans)

# Read the number of test cases
T = int(sys.stdin.readline())

for _ in range(T):
    solve()