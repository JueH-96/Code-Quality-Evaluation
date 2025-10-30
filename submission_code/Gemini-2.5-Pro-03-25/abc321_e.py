# YOUR CODE HERE
import sys

# Function to calculate count of nodes in range [L, R] that are <= N
# It computes max(0, min(R, N) - L + 1)
# Assumes L and R can be very large, N is up to 10^18
def count_nodes_in_range(L, R, N):
    """
    Counts the number of integers v such that L <= v <= R and v <= N.
    L, R, N can be large integers (up to 10^18 or more).
    Python's arbitrary precision integers handle large values.
    """
    # If the start of the range is already greater than N, no nodes are valid.
    if L > N:
        return 0
    
    # The number of integers in [L, R] is R - L + 1.
    # We are interested in integers v such that L <= v <= R and v <= N.
    # This is equivalent to L <= v <= min(R, N).
    # The number of integers in this range is min(R, N) - L + 1.
    # Since we established L <= N, and L <= R (by construction of ranges in this problem),
    # we know min(R, N) >= L. Thus min(R, N) - L + 1 >= 1.
    # The use of max(0, ...) is just a safeguard, not strictly necessary here.
    # return max(0, min(R, N) - L + 1) # max(0, ...) is redundant if logic is correct.
    return min(R, N) - L + 1

def solve():
    """
    Solves a single test case. Reads N, X, K and computes the answer.
    """
    N, X, K = map(int, sys.stdin.readline().split())

    ans = 0

    # Handle K=0 separately. The only vertex at distance 0 from X is X itself.
    if K == 0:
        print(1) 
        return

    # Initialize total count
    ans = 0

    # Case 1: Descendants of X at distance K (applies only if K >= 1)
    # The descendants of X at distance K form a range of vertex IDs.
    # The range is [X * 2^K, (X+1) * 2^K - 1]. Let's call it [L_desc, R_desc].
    # We need to count how many vertices v in this range satisfy 1 <= v <= N.
    # Since X >= 1, K >= 1, L_desc = X * 2^K >= 1 * 2^1 = 2. So v >= 1 is always satisfied.
    # We only need to check v <= N.
    
    try:
        # Heuristic optimization: if K is very large (e.g., >= 70), 2^K is huge.
        # For N <= 10^18 (approx 2^60), X * 2^K will likely exceed N unless X=0 (not allowed).
        # If K is large, computing 2^K might be slow or memory intensive unnecessarily.
        # A threshold like 70 covers beyond 64-bit range comfortably.
        if K >= 70: 
             # If K is very large, L_desc = X * 2^K will definitely be > N.
             # So the count for descendants is 0. Skip computation.
             pass 
        else: # K is reasonably small
            pow2K = 1 << K
            lower_desc = X * pow2K
            
            # Check potential overflow with X+1? Python handles large integers.
            # upper_desc = (X + 1) * pow2K - 1
            # Simplified calculation for upper bound: lower_desc + pow2K - 1
            # Be careful if X+1 wraps around max int? Not in Python.
            # Let's use (X+1) * 2^K - 1 for clarity. It is equivalent to lower_desc + (1 << K) - 1
            upper_desc = lower_desc + pow2K - 1
            
            # Add count of nodes in the valid range [lower_desc, upper_desc] that are <= N.
            ans += count_nodes_in_range(lower_desc, upper_desc, N)

    except OverflowError: 
         # This block is unlikely to be reached with Python's arbitrary precision integers.
         # If it somehow occurred, it implies numbers are extremely large, count is 0.
         pass 

    # Case 2: Non-descendant vertices at distance K.
    # These vertices are reached by going up k_up steps from X to an ancestor L,
    # then going down k_down = K - k_up steps into a different subtree rooted at L.
    
    curr = X # Start at vertex X
    
    # Iterate upwards towards the root, step by step.
    # k_up tracks the distance from X to the current ancestor `parent`.
    for k_up in range(1, K + 1):
        parent = curr // 2 # Calculate the parent of the current node
        
        # If parent is 0, we have climbed past the root (vertex 1). Stop.
        if parent == 0: 
            break

        # Calculate remaining distance required downwards
        k_down = K - k_up

        # Case 2a: The ancestor `parent` itself is the target vertex.
        # This happens when k_down = 0, meaning K = k_up.
        if k_down == 0:
            # The vertex `parent` is at distance K from X.
            # Count this ancestor vertex. Since parent != 0, parent >= 1.
            ans += 1
            # Note: loop could continue if K is larger than depth(X), but k_down=0 case only happens once.
        
        # Case 2b: The target vertex is in the subtree of `parent` but not `curr`.
        # This happens when k_down > 0.
        elif k_down > 0:
            # Find the sibling `S` of `curr`. `curr` is the child of `parent` on the path towards X.
            # The target vertex must be in the subtree rooted at `S`.
            S = 0
            # Determine sibling S based on whether curr is left (even) or right (odd) child.
            if curr % 2 == 0: # curr = 2 * parent (left child)
                S = parent * 2 + 1 # Sibling S is the right child
            else: # curr = 2 * parent + 1 (right child)
                S = parent * 2 # Sibling S is the left child

            # Only proceed if the sibling S exists within the N vertices.
            if S <= N:
                # We need to find descendants of S at distance k_dist_from_S = k_down - 1.
                k_dist_from_S = k_down - 1
                
                # Check if distance is valid (non-negative)
                if k_dist_from_S >= 0: 
                    try:
                        # Similar heuristic check for large k_dist_from_S
                        if k_dist_from_S >= 70:
                             # If k_dist_from_S is large, S * 2^k_dist_from_S likely exceeds N.
                             # Count is 0. Skip computation.
                             pass
                        else:
                            # Calculate the range [L_S, R_S] for descendants of S at this distance.
                            pow2_k_dist = 1 << k_dist_from_S
                            lower_S = S * pow2_k_dist
                            # upper_S = (S + 1) * pow2_k_dist - 1
                            upper_S = lower_S + pow2_k_dist - 1 # Equivalent calculation
                            
                            # Add the count of valid nodes in this range [lower_S, upper_S] that are <= N.
                            ans += count_nodes_in_range(lower_S, upper_S, N)
                    except OverflowError:
                        # Safeguard against potential overflow.
                        pass 

        # Move one step up for the next iteration: current node becomes the parent.
        curr = parent
        
    # Print the final total count.
    print(ans)

# Read the number of test cases.
T = int(sys.stdin.readline())
# Process each test case.
for _ in range(T):
    solve()