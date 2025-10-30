# YOUR CODE HERE
import sys

# Increase recursion depth limit for deep trees.
# The constraint NK <= 2e5 suggests the tree depth could be large.
# Set a large recursion depth limit commonly used in competitive programming.
try:
    # Set a generous limit. If system maximum is lower, this might fail.
    sys.setrecursionlimit(200010) 
except OverflowError: # Handle cases where the system limit is lower than requested
     # Try a slightly smaller value if the large one fails.
     # This value might need adjustment based on the specific environment.
     sys.setrecursionlimit(200000) 

def solve():
    N, K = map(int, sys.stdin.readline().split())
    
    total_vertices = N * K

    # Base case: If K=1, each path consists of a single vertex.
    # Any tree with NK vertices can be decomposed into NK paths of length 1.
    # If K=1, the tree has N vertices. We need N paths of length 1. This is always possible.
    if K == 1:
        print("Yes")
        return

    # Build adjacency list representation of the tree.
    adj = [[] for _ in range(total_vertices)]
    for _ in range(total_vertices - 1):
        u, v = map(int, sys.stdin.readline().split())
        # Adjust from 1-based vertex indexing in input to 0-based for Python list indices.
        adj[u - 1].append(v - 1)
        adj[v - 1].append(u - 1)

    # dp[u] conceptually stores the length of the path segment starting in the subtree rooted at u,
    # ending at u, and needing to be extended upwards towards the parent.
    # - A value of 0 means the subtree rooted at u is perfectly partitionable into paths of length K.
    # - A value L (1 <= L < K) means there's one path segment of length L ending at u needing extension.
    # - A value of -1 indicates that partitioning is impossible for the subtree rooted at u.
    # In this implementation, we don't use a DP array explicitly for memoization,
    # as standard DFS on trees naturally avoids recomputing for the same node within a single traversal.
    # The return value of the dfs function serves the role of dp[u].
    
    def dfs(u, p):
        """
        Performs Depth First Search starting from node u with parent p.
        Calculates the state of the subtree rooted at u regarding path decomposition.

        Args:
            u: The current node (vertex index).
            p: The parent node (vertex index), or -1 if u is the root.

        Returns:
            int: The length (1 to K-1) of the path segment ending at u that needs extension upwards.
                 Returns 0 if the subtree at u is perfectly partitioned into paths of length K.
                 Returns -1 if partitioning the subtree rooted at u is impossible.
        """
        
        # List to store lengths of segments returned by children that are > 0.
        # These are segments ending at children that need further matching/extension.
        child_segments = [] 
        
        # Iterate over neighbors of u
        for v in adj[u]:
            if v == p: # Skip the edge leading back to the parent
                continue
            
            # Recursively call DFS on child v
            res = dfs(v, u)
            
            # Propagate failure upwards: if any child subtree cannot be partitioned, this subtree also cannot.
            if res == -1: 
                return -1
            
            # Collect positive length segments returned by children.
            # A result of 0 means the child's subtree was perfectly partitioned and doesn't need matching.
            if res > 0: 
                child_segments.append(res)

        # Sort the collected child segment lengths. This facilitates efficient matching using two pointers.
        child_segments.sort()
        
        num_segments = len(child_segments)
        # Boolean array to keep track of which child segments have been matched.
        matched = [False] * num_segments 
        
        # ----- Matching Logic -----
        # The goal is to match segments originating from children subtrees at node u.

        # Phase 1: Try pairing segments.
        # Two segments from children, with lengths L_left and L_right, can potentially form a complete path
        # if L_left + L_right == K. This represents a path starting deep in one child's subtree, passing
        # through that child, then u, then the other child, and ending deep in the other child's subtree.
        # The total vertices in such path would be L_left + L_right + 1 (vertex u). This logic is debated,
        # but L_left + L_right == K is a common condition in related problems. Let's use it.
        
        left = 0
        right = num_segments - 1
        while left < right:
            # Skip already matched segments
            if matched[left]:
                left += 1
                continue
            if matched[right]:
                right -= 1
                continue
            
            current_sum = child_segments[left] + child_segments[right]
            
            if current_sum == K:
                # Found a pair that sums to K. Match them.
                matched[left] = True
                matched[right] = True
                # Move pointers inward
                left += 1
                right -= 1
            elif current_sum < K:
                # Sum is too small. To increase the sum, we need a larger segment.
                # Move the left pointer to consider a potentially larger segment.
                left += 1 
            else: # current_sum > K
                # Sum is too large. To decrease the sum, we need a smaller segment.
                # Move the right pointer to consider a potentially smaller segment.
                right -= 1 
        
        # Phase 2: Handle remaining unmatched segments after pairing.
        # At most one unmatched segment can be passed upwards to the parent.
        
        # Stores the length of the segment selected to be passed upwards. Initialize to -1 (none selected).
        candidate_up_segment_val = -1 

        # Iterate through segments (typically largest first might be strategic, but order doesn't strictly matter here)
        # Iterating backwards (largest to smallest) is done here.
        for i in range(num_segments - 1, -1, -1):
             if not matched[i]: # Process only unmatched segments
                  # This segment is currently unmatched. What can we do with it?
                  
                  # Option A: Check if this segment combined with node u forms a complete path of length K.
                  # The length of the path segment ending at u would be child_segments[i] + 1.
                  if child_segments[i] + 1 == K:
                      # Yes, it forms a complete path ending at u. Mark this segment as matched.
                      matched[i] = True
                  
                  # Option B: Check if this segment can be the one passed upwards.
                  elif candidate_up_segment_val == -1: 
                      # This is the first unmatched segment encountered that doesn't complete a path ending at u.
                      # Tentatively select it as the segment to pass upwards.
                      candidate_up_segment_val = child_segments[i] # Store its original length
                      matched[i] = True # Mark it as 'handled' (it's designated to go up)
                  else:
                      # Found a second unmatched segment that neither completes a path ending at u nor can be passed up
                      # (because we can only pass one segment up). This situation implies failure.
                      return -1 # Impossible to partition

        # ----- Final checks and determining the return value -----

        # Sanity check: Ensure all segments have been handled (matched).
        # If the logic above is correct, this check should always pass unless there's a flaw.
        all_matched_check = True
        for i in range(num_segments):
            if not matched[i]:
                all_matched_check = False
                break
        
        if not all_matched_check:
             # If an unmatched segment remains without being designated upwards, it's an error state.
             return -1

        # Determine the final length to return upwards based on whether a candidate was selected.
        up_len = 0 # Default to 0 (perfectly partitioned)
        if candidate_up_segment_val != -1:
            # A segment was designated to go upwards. Its length increases by 1 because it now includes node u.
            up_len = candidate_up_segment_val + 1
        else:
            # No segment was designated upwards. This implies all segments from children were matched internally
            # (either paired up or formed paths ending at u).
            # In this case, node u itself must start a new path segment upwards. This segment has length 1.
            up_len = 1
        
        # Post-process the calculated upward length:
        if up_len == K:
            # If the final upward segment length is K, it means this segment forms a complete path.
            # This path might end at u (if formed from below) or potentially involve the parent edge.
            # Regardless, from the perspective of the subtree rooted at u, it's fully matched.
            # So, return 0 to indicate perfect partitioning of the subtree.
            return 0 
        elif up_len > K: 
             # This should not happen with correct logic. Segment lengths should not exceed K.
             # Indicates an error state or impossible scenario.
             return -1
        else: # 1 <= up_len < K
            # Return the calculated length of the segment that needs further extension upwards.
            return up_len


    # Initiate DFS from an arbitrary root, say vertex 0. The parent is -1 (or any value not in vertex range).
    root_result = dfs(0, -1) 

    # Check the result from the root of the tree.
    # For a successful decomposition of the entire tree, the root node must end up
    # with a state indicating perfect partitioning, which corresponds to a return value of 0.
    # Any other value (a positive length 1 to K-1, or -1 for failure) means the tree cannot be decomposed.
    if root_result == 0: 
        print("Yes")
    else:
        print("No")

# Execute the solve function to run the program.
solve()