# YOUR CODE HERE
import sys

def solve():
    # Read N, A, B from input. N is the number of plans.
    # A is the number of holidays at the start of the week.
    # B is the number of weekdays following the holidays.
    N, A, B = map(int, sys.stdin.readline().split())
    
    # Read the list of D_i values, representing the number of days later each plan is scheduled.
    # The problem statement guarantees D_1 < D_2 < ... < D_N.
    D = list(map(int, sys.stdin.readline().split()))

    # Calculate the total length of the week.
    W = A + B

    # Calculate the remainders of D_i when divided by W. This gives the day of the week
    # each plan falls on, assuming today is day 0 (0-based indexing).
    # We use a set to automatically handle duplicate remainders and keep only unique ones.
    rems = set()
    for x in D:
        rems.add(x % W)

    # Convert the set of unique remainders to a sorted list.
    # Let these unique remainders be u_1, u_2, ..., u_m.
    unique_rems = sorted(list(rems))
    
    # Get the number of unique remainders.
    m = len(unique_rems)

    # If there are no unique remainders (e.g., if N=0 was allowed by constraints),
    # then any starting day works. The logic below correctly handles the case m=1.
    # Since N >= 1, m >= 1 is guaranteed.

    # We need to determine if there exists a day k' (0 <= k' < W) such that for all plans i,
    # the day (k' + D_i) mod W falls within the holiday range [0, A-1].
    # This is equivalent to finding if there exists a k' such that for all unique remainders u_j,
    # (k' + u_j) mod W is in [0, A-1].
    # This condition holds if and only if the span of the set of unique remainders {u_j} on the circle
    # is less than or equal to A-1.

    # Calculate the maximum gap between consecutive unique remainders when arranged on a circle.
    # The maximum gap represents the largest interval on the circle that contains *no* plan days (modulo W).
    max_gap = 0

    if m > 0: # This condition is always true since N >= 1 guarantees m >= 1.
        
        # Calculate gaps between adjacent elements in the sorted list unique_rems.
        # Example: if unique_rems = [u_1, u_2, u_3], gaps are (u_2 - u_1) and (u_3 - u_2).
        for i in range(m - 1):
            # Gap is the difference between consecutive sorted remainders.
            gap = unique_rems[i+1] - unique_rems[i]
            # Update the maximum gap found so far.
            if gap > max_gap:
                 max_gap = gap
                 
        # Calculate the wrap-around gap: distance from the last element u_m back to the first element u_1.
        # On the circle of length W, this distance is (u_1 + W) - u_m.
        wrap_around_gap = unique_rems[0] + W - unique_rems[m-1]
        # Update max_gap if the wrap-around gap is larger.
        if wrap_around_gap > max_gap:
            max_gap = wrap_around_gap
            
    else: # This case m=0 is not possible because N >= 1 constraint.
         # If it were possible (e.g. N=0), span should be 0. Setting max_gap = W ensures span = W - W = 0.
         max_gap = W

    # The span S is the length of the minimum arc covering all unique remainder points {u_j}.
    # It is calculated as W - maximum_gap.
    # The span represents the minimum length required to contain all relative plan days on the circle.
    span = W - max_gap

    # Check if the minimum required arc length (span) fits within the available holiday range length.
    # The holiday days are 0-indexed from 0 to A-1. The length of this interval is A.
    # The maximum possible difference between two points within [0, A-1] is (A-1) - 0 = A-1.
    # If the span S of the points {u_j} is <= A-1, then there exists a rotation (choice of starting day k')
    # such that all rotated points (k' + u_j) mod W fall within an interval of length S starting at 0,
    # i.e., [0, S]. Since S <= A-1, this interval [0, S] is contained within the holiday interval [0, A-1].
    # Therefore, it is possible if and only if span <= A-1.
    if span <= A - 1:
        print("Yes")
    else:
        print("No")

# Execute the solve function to read input, process, and print output.
solve()