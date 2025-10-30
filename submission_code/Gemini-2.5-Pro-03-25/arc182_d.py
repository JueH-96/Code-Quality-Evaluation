# YOUR CODE HERE
import sys

def dist(u, v, M):
    """
    Calculates the shortest distance between u and v on a cycle of size M.
    The distance is the minimum number of +1 or -1 operations modulo M
    to get from u to v.
    """
    # Clockwise distance: (v - u) mod M
    diff_cw = (v - u + M) % M
    if diff_cw == 0:
        return 0
    # Counter-clockwise distance: (u - v) mod M, which is also M - diff_cw
    diff_ccw = M - diff_cw 
    return min(diff_cw, diff_ccw)

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Handle the special case M=2
    if M == 2:
        # A sequence is "good" if no two adjacent elements are the same.
        # For M=2, this means adjacent elements must be 0 and 1 (or 1 and 0).
        # Check if A and B are identical sequences.
        if A == B:
             print(0) # If they are identical, 0 operations are needed.
        else:
             # If A != B, transformation is impossible.
             # Any operation (+1 or -1 mod 2) on an element A_i will make it equal to its neighbor(s),
             # because the neighbor must have the different value (0 vs 1).
             # For example, if A = (0, 1), changing A[0] using (+1 mod 2) results in (1, 1). Not good.
             # Changing A[0] using (-1 mod 2) results in (1, 1). Not good.
             # Similarly for A[1].
             # Therefore, if A and B are different good sequences with M=2, it's impossible to transform A to B.
             print("-1")
        return

    # Initialize lists for signed displacement and distance for each element
    s = [0] * N # s[i] stores the signed displacement for A[i] -> B[i] along shortest path
                # Positive means clockwise, negative means counter-clockwise.
    k = [0] * N # k[i] stores the distance d(A[i], B[i]), which is |s[i]|.
    
    total_k = 0 # Stores the sum of shortest distances, the base minimum operations if no constraints existed.
    
    # Calculate distances and determine initial signed displacements for each element
    for i in range(N):
        # Calculate clockwise distance from A[i] to B[i]
        diff_cw = (B[i] - A[i] + M) % M
        
        ki = 0 # Distance k[i]
        if diff_cw == 0:
           # If A[i] == B[i], distance is 0, displacement is 0.
           ki = 0
           s[i] = 0
        # Check if clockwise path is shorter or equal length compared to counter-clockwise path
        # The counter-clockwise distance is M - diff_cw.
        # So clockwise is shorter/equal if diff_cw <= M - diff_cw, which simplifies to diff_cw <= M / 2.
        elif diff_cw <= M / 2.0: 
           ki = diff_cw
           s[i] = ki # Positive displacement indicates clockwise path is shortest (or equal if M/2)
        else: # Counter-clockwise path is shorter
           ki = M - diff_cw # The distance is the counter-clockwise length
           s[i] = -ki # Negative displacement indicates counter-clockwise path is shortest
        
        # Store distance and add to total base cost
        k[i] = ki
        total_k += ki

    # Initialize total extra cost accumulated due to resolving deadlocks
    total_extra_cost = 0
    
    # Iterate through adjacent pairs (A[i], A[i+1]) to detect and handle deadlocks
    for i in range(N - 1):
        hard_deadlock = False # Flag to indicate a deadlock that requires extra cost
        
        # Check for potential deadlock Case 1: A[i+1] is one step clockwise from A[i]
        if A[i+1] == (A[i] + 1) % M:
             # A deadlock occurs if A[i] needs to move clockwise (s[i]>0) to reach B[i]
             # AND A[i+1] needs to move counter-clockwise (s[i+1]<0) to reach B[i+1].
             # This causes them to potentially collide or block each other's first step.
             if s[i] > 0 and s[i+1] < 0:
                 # This potential deadlock requires extra cost only if both paths are *unique* shortest paths.
                 # A path is unique if its length k is strictly less than M/2.
                 # If k = M/2 (only possible if M is even), there are two shortest paths,
                 # allowing a choice of direction that might avoid the deadlock without extra cost.
                 is_i_unique = (k[i] < M / 2.0) # Use float division for comparison
                 is_i1_unique = (k[i+1] < M / 2.0)
                 if is_i_unique and is_i1_unique:
                     hard_deadlock = True # Both paths unique and conflicting, hard deadlock
        
        # Check for potential deadlock Case 2: A[i+1] is one step counter-clockwise from A[i]
        elif A[i+1] == (A[i] - 1 + M) % M:
             # A deadlock occurs if A[i] needs to move counter-clockwise (s[i]<0)
             # AND A[i+1] needs to move clockwise (s[i+1]>0).
             if s[i] < 0 and s[i+1] > 0:
                  # Check if both paths are unique shortest paths
                 is_i_unique = (k[i] < M / 2.0)
                 is_i1_unique = (k[i+1] < M / 2.0)
                 if is_i_unique and is_i1_unique:
                     hard_deadlock = True # Both paths unique and conflicting, hard deadlock
        
        # If a hard deadlock is detected, calculate the minimum extra cost needed to resolve it
        if hard_deadlock:
            # Calculate the extra cost if A[i] takes a detour step (opposite direction of its shortest path)
            if s[i] > 0: # If A[i]'s shortest path is clockwise, detour step is counter-clockwise
                Ai_detour_val = (A[i] - 1 + M) % M
            else: # If A[i]'s shortest path is counter-clockwise, detour step is clockwise
                Ai_detour_val = (A[i] + 1) % M
            
            # Calculate the shortest distance from the detour value (Ai_detour_val) to the target B[i]
            d_Ai_detour_Bi = dist(Ai_detour_val, B[i], M)
            # Calculate the extra cost for A[i] detour:
            # Cost = (1 step for detour + new path length from detour value) - original path length
            delta_Ci = 1 + d_Ai_detour_Bi - k[i]

            # Calculate the extra cost if A[i+1] takes a detour step
            if s[i+1] > 0: # If A[i+1]'s path is clockwise, detour is counter-clockwise
                Ai1_detour_val = (A[i+1] - 1 + M) % M
            else: # If A[i+1]'s path is counter-clockwise, detour is clockwise
                Ai1_detour_val = (A[i+1] + 1) % M

            # Calculate the shortest distance from the detour value (Ai1_detour_val) to the target B[i+1]
            d_Ai1_detour_Bi1 = dist(Ai1_detour_val, B[i+1], M)
            # Calculate the extra cost for A[i+1] detour
            delta_Ci1 = 1 + d_Ai1_detour_Bi1 - k[i+1]
            
            # To resolve the deadlock, only one element needs to detour initially.
            # We choose the detour that incurs the minimum extra cost.
            total_extra_cost += min(delta_Ci, delta_Ci1)

    # The final minimum number of operations is the sum of base costs plus total extra costs from deadlocks
    print(total_k + total_extra_cost)

solve()