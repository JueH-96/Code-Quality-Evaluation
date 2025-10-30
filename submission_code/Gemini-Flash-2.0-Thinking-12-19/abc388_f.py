import sys
from collections import deque

# Use input() for reading from stdin
input = sys.stdin.readline

def solve():
    N, M, A, B = map(int, input().split())
    
    L = []
    R = []
    for _ in range(M):
        l, r = map(int, input().split())
        L.append(l)
        R.append(r)

    # Case 1: A == B
    if A == B:
        # We can only move by step A.
        # N is reachable if and only if N = 1 + kA for some integer k >= 0,
        # and all intermediate squares 1 + jA (0 <= j <= k) are good.
        # Since 1 is always good and R_i < N, N is always good.
        # The squares on the path are 1, 1+A, 1+2A, ..., 1 + (N-1)/A * A.
        # This requires (N - 1) to be divisible by A for the path to potentially end exactly at N.
        if (N - 1) % A != 0:
            print("No")
            return
        
        # The squares on the path are 1 + kA for k = 0, 1, ..., (N-1)//A.
        # We need to check if any square 1 + kA for k = 1, ..., (N-1)//A is bad.
        # A square `pos` is bad if L_i <= pos <= R_i for some i.
        # Iterate through bad intervals [L_i, R_i]. Check if any square 1 + kA (k >= 1) falls into [L_i, R_i].
        
        for i in range(M):
            l_i, r_i = L[i], R[i]
            
            # We are looking for integer k >= 1 such that l_i <= 1 + k*A <= r_i.
            # This is equivalent to l_i - 1 <= k*A <= r_i - 1, with k >= 1.
            
            # Smallest integer k >= 1 such that k*A >= l_i - 1.
            # Since l_i > 1, l_i - 1 >= 1.
            # Need k >= (l_i - 1) / A. Smallest integer k >= 1 is max(1, ceil((l_i - 1)/A)).
            # Using integer division for ceil(x/y) where x > 0, y > 0 is (x + y - 1) // y.
            # So ceil((l_i - 1)/A) is (l_i - 1 + A - 1) // A.
            k_start = max(1, (l_i - 1 + A - 1) // A) if l_i > 1 else 1 # L_i > 1 is guaranteed

            # Largest integer k >= 1 such that k*A <= r_i - 1.
            # k <= (r_i - 1) / A. Largest integer k is floor((r_i - 1) / A).
            k_end = (r_i - 1) // A if r_i >= 1 else -1 # If r_i < 1, no k >= 0. But r_i >= L_i > 1.

            # We need to check if there exists an integer k such that k_start <= k <= k_end.
            # If such a k exists, then 1 + k*A is in [L_i, R_i] and k >= k_start >= 1.
            # Since R_i < N, 1 + k*A <= R_i < N, so this square is < N.
            # The path goes up to k = (N-1)//A. We need to check if 1 + k_start * A is on this path, i.e., k_start <= (N-1)//A.
            # Since R_i < N, R_i - 1 < N - 1. If A > 0, (R_i - 1) // A <= (N - 1) // A.
            # Thus, k_end <= (N - 1) // A.
            # If k_start <= k_end, then k_start <= (N-1)//A is implicitly satisfied if k_start >= 1.
            # So, we just need to check if k_start <= k_end.
            
            if k_start <= k_end:
                # There exists an integer k in the range [k_start, k_end] which is also >= 1.
                # The square 1 + k*A is bad and is on the path 1, 1+A, ..., N.
                print("No")
                return

        # If the loop completes, no bad square was found on the sequence 1, 1+A, ..., N.
        # And we already checked that N is reachable exactly (N-1)%A == 0.
        print("Yes")
        return

    # Case 2: A < B
    
    # Helper function to merge a list of intervals into a sorted list of disjoint intervals.
    def merge_intervals(intervals):
        if not intervals:
            return []
        # Sort intervals by start point.
        # intervals = sorted(intervals) # Already sorted if appended correctly, but safer to sort.
        intervals.sort() 
        
        merged = []
        if not intervals: return []
        
        current_merged = list(intervals[0]) # Use list to allow modification
        merged.append(current_merged)
        
        for next_interval in intervals[1:]:
            if next_interval[0] <= current_merged[1] + 1:
                # Overlap or adjacent, merge by extending the end
                current_merged[1] = max(current_merged[1], next_interval[1])
            else:
                # No overlap, add the next interval as a new one
                current_merged = list(next_interval) # Use list to allow modification
                merged.append(current_merged)
        return merged

    # Helper function to extract good portions within a range [start, end].
    # Assumes L, R are sorted lists of bad interval boundaries, M is count, N is total squares.
    def get_good_parts(start, end, L, R, M, N):
        result = []
        
        # Clip the range [start, end] to be within [1, N]
        effective_start = max(1, start)
        effective_end = min(end, N)

        if effective_start > effective_end:
            return []

        pos = effective_start
        bad_i = 0 # Index for the bad_intervals lists L and R
        
        while pos <= effective_end:
            # Advance bad_i to the first interval [L[bad_i], R[bad_i]] where R[bad_i] >= pos.
            # Any interval before this is completely to the left of pos.
            while bad_i < M and R[bad_i] < pos:
                bad_i += 1
            
            # If we are past all bad intervals or the current one starts after our range ends
            if bad_i >= M or L[bad_i] > effective_end: # If L[bad_i] > effective_end, the bad interval starts strictly after the range [pos, effective_end].
                 # The segment [pos, effective_end] is good.
                 result.append([pos, effective_end])
                 pos = effective_end + 1 # Finished processing this range
                 break # Exit the while loop
            
            # We are at pos, and the relevant bad interval starts at L[bad_i].
            L_bad, R_bad = L[bad_i], R[bad_i]

            # If pos is before the bad interval starts [L_bad, R_bad]
            # The segment [pos, min(effective_end, L_bad - 1)] is good.
            if pos < L_bad:
                good_segment_end = min(effective_end, L_bad - 1)
                
                # Add the good segment. This segment is guaranteed valid as pos < L_bad.
                result.append([pos, good_segment_end])
                
                # Move pos past this good segment.
                pos = good_segment_end + 1
                # Do not increment bad_i yet, the next iteration will re-evaluate pos against the same bad interval.
                
            # If pos is inside or at the start of the bad interval [L_bad, R_bad].
            # This happens if pos >= L_bad, and given the while R[bad_i] < pos condition above,
            # it means pos <= R[bad_i]. So pos is in [L_bad, R_bad].
            elif pos <= R_bad: # This condition captures pos >= L_bad and pos <= R_bad
                # Skip the bad interval. The next potential good square is R_bad + 1.
                pos = R_bad + 1
                # Do not increment bad_i yet. The next iteration will re-evaluate pos.

            # If pos > R_bad, the initial `while R[bad_i] < pos` should have advanced bad_i.
            # This branch should not be reached in a correct flow.
            # If it is reached, it means pos > R_bad, so the current bad interval is behind us.
            # We should just continue the loop, the next iteration will advance bad_i.
            # continue # Redundant, happens naturally by while loop condition

    return result


    # Main loop for A < B case
    # Reachable set represented as a sorted list of disjoint intervals.
    reachable = [[1, 1]] 
    last_reachable = [] # To check for termination

    while reachable != last_reachable:
        last_reachable = reachable
        
        newly_generated_intervals = []
        
        # Iterate through current reachable intervals
        for s, e in reachable:
            # Calculate the potential next reachable range [s+A, e+B]
            potential_start = s + A
            potential_end = e + B
            
            # Find the good parts within this potential range, clipped by N
            newly_generated_intervals.extend(get_good_parts(potential_start, potential_end, L, R, M, N))

        # Merge newly generated intervals with the existing reachable intervals
        # This step updates the `reachable` set.
        reachable = merge_intervals(reachable + newly_generated_intervals)
        
        # After merging, check if N is now reachable
        for s, e in reachable:
            if s <= N <= e:
                print("Yes")
                return

    # If the loop finishes and N is not reachable
    print("No")

# Read input and solve
solve()