# YOUR CODE HERE
import sys

# Function to read input and solve the problem
def solve():
    # Read N (number of islands/bridges) and M (number of islands in the tour sequence) from input
    N, M = map(int, sys.stdin.readline().split())
    
    # Read the sequence of visited islands X
    X = list(map(int, sys.stdin.readline().split()))

    # Initialize the total length of the tour in the original graph (without any bridge closure)
    L0 = 0
    
    # Initialize the difference array for tracking the potential increase in path length
    # if a specific bridge is closed. Size N+2 allows using 1-based indexing for bridges 1 to N,
    # plus an index N+1 for handling range updates wrapping around the cycle.
    # diff_increase[i] stores the change in the 'increase' value starting from bridge i.
    diff_increase = [0] * (N + 2) 

    # Iterate through consecutive pairs of islands (X_k, X_{k+1}) in the tour sequence.
    # There are M-1 such segments in the tour.
    for k in range(M - 1):
        # Current segment is from island u to island v.
        u, v = X[k], X[k+1]
        
        # Calculate the lengths of the two paths between u and v in the cycle graph C_N.
        # The islands are numbered 1 to N. The bridges are (1,2), (2,3), ..., (N-1, N), (N,1).
        
        # dist1: length of the path following island indices sequentially (e.g., 1->2->3 or 3->2->1).
        # This path does not use the bridge (N,1) if we consider indices linearly.
        dist1 = abs(u - v)
        
        # dist2: length of the path that wraps around using the (N,1) bridge.
        dist2 = N - dist1
        
        # The shortest path distance 'd' in the original cycle graph is the minimum of these two path lengths.
        d = min(dist1, dist2)
        
        # Add this shortest distance to the total initial tour length L0.
        L0 += d
        
        # If dist1 == dist2, it means d = N/2 (N must be even). In this case, there are two shortest paths
        # of equal length between u and v. Closing a bridge on one path still leaves the other shortest path available.
        # Therefore, the shortest distance doesn't change, and there's no increase in length for this segment.
        # We only need to consider updates if dist1 != dist2 (i.e., d < N/2), which implies a unique shortest path exists.
        if dist1 != dist2: 
            # Calculate the increase in path length for this segment if a bridge on its unique shortest path is closed.
            # If the shortest path is blocked, the tour must take the alternative path.
            # The new path length would be the length of the longer path, which is N-d.
            # The increase in length for this segment is (N-d) - d = N - 2*d.
            val = N - 2 * d
            
            # Determine which path is the unique shortest one and update the difference array accordingly.
            # The goal is to add 'val' to the 'increase' count for every bridge that lies on this shortest path.
            
            # Case 1: The path not crossing the N->1 connection is shorter (dist1 < dist2).
            # The bridges on this path connect islands min(u,v) through max(u,v).
            # The bridge indices are lower, lower+1, ..., upper-1 where lower=min(u,v), upper=max(u,v).
            if dist1 < dist2:
                lower = min(u, v)
                upper = max(u, v)
                # Add 'val' to the range [lower, upper-1].
                # Using difference array: increment 'val' at the start index 'lower', 
                # and decrement 'val' at the index immediately after the end ('upper').
                diff_increase[lower] += val
                diff_increase[upper] -= val
            
            # Case 2: The path crossing the N->1 connection is shorter (dist2 < dist1).
            # The bridges on this path include bridge N (connecting N and 1) and potentially others.
            # The set of bridge indices is {1, ..., lower-1} U {upper, ..., N}, where lower=min(u,v), upper=max(u,v).
            else: # dist2 < dist1
                lower = min(u, v)
                upper = max(u, v)
                
                # Add 'val' to the range [1, lower-1].
                # Apply difference array update: add 'val' at index 1, subtract 'val' at index 'lower'.
                # If lower=1, this range is empty and the updates cancel out.
                diff_increase[1] += val
                diff_increase[lower] -= val
                
                # Add 'val' to the range [upper, N].
                # Apply difference array update: add 'val' at index 'upper', subtract 'val' at index N+1.
                # The decrement at N+1 ensures the effect is limited to indices up to N.
                diff_increase[upper] += val
                diff_increase[N+1] -= val 

    # Calculate the actual total increase in tour length if bridge 'i' is closed.
    # This is done by computing the prefix sums of the difference array.
    # While doing so, find the minimum total increase among all possible bridges (1 to N).
    min_increase = float('inf')
    current_increase = 0
    for i in range(1, N + 1): # Iterate through bridge indices 1 to N
        # Update the current total increase by adding the change starting at bridge i.
        current_increase += diff_increase[i]
        # Keep track of the minimum total increase found so far.
        min_increase = min(min_increase, current_increase)

    # If min_increase remained infinity, it means the difference array was all zeros.
    # This could happen if M=1 (no segments) or if all segments had length N/2.
    # With constraints N>=3, M>=2, min_increase should always receive a finite value (possibly 0).
    # The check is mostly a safeguard. If no increases occurred, min_increase should be 0.
    if min_increase == float('inf'):
         min_increase = 0 

    # The minimum possible total tour length after closing one optimally chosen bridge is
    # the original tour length L0 plus the minimum possible increase found.
    print(L0 + min_increase)

# Call the solve function to execute the program logic
solve()