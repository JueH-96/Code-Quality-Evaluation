import collections
import sys

# It's good practice to increase recursion limit if needed, though not for BFS.
# sys.setrecursionlimit(10**6)

# Fast I/O can be helpful for large inputs, though often not the bottleneck in Python.
# input = sys.stdin.readline 

def solve():
    N, M, A, B = map(int, sys.stdin.readline().split())
    
    bad_intervals = []
    if M > 0:
        for _ in range(M):
            l, r = map(int, sys.stdin.readline().split())
            bad_intervals.append((l, r))
        # bad_intervals are already sorted by L_i and non-overlapping as per constraints.

    if M == 0:
        target_dist = N - 1
        if target_dist < A: # Cannot make any move if N-1 is less than smallest step A
            print("No")
            return

        if A == B: # Only steps of size A are possible
            if target_dist % A == 0:
                print("Yes")
            else:
                print("No")
            return

        # A < B. Any sufficiently large distance is formable.
        # Max Frobenius number for A, B <= 20 is for {19, 20}, which is 341.
        # Using a threshold like 400 is safe.
        FROBENIUS_THRESHOLD = 400 
        
        if target_dist > FROBENIUS_THRESHOLD:
            print("Yes")
            return

        dp_formable = [False] * (target_dist + 1)
        for i in range(A, B + 1):
            if i <= target_dist:
                dp_formable[i] = True
        
        for k_dist in range(A + 1, target_dist + 1): # Start from A+1, check smaller sums
            if dp_formable[k_dist]: # Already known (e.g. k_dist itself is in [A,B])
                continue
            for step_val in range(A, B + 1):
                if k_dist - step_val >= 0 and dp_formable[k_dist - step_val]:
                    dp_formable[k_dist] = True
                    break # Found a way to form k_dist
        
        if dp_formable[target_dist]:
            print("Yes")
        else:
            print("No")
        return

    # M > 0 case
    # BFS
    q = collections.deque()
    # Square 1 is guaranteed not to be bad by constraints (1 < L_i)
    q.append(1)
    visited = {1}

    while q:
        curr_sq = q.popleft()

        if curr_sq == N:
            print("Yes")
            return

        for step in range(A, B + 1):
            next_sq = curr_sq + step
            
            if next_sq > N:
                continue
            
            if next_sq in visited:
                continue
            
            # Check if next_sq is bad using binary search on bad_intervals
            is_next_sq_bad = False
            # Find first interval [L_k, R_k] such that L_k <= next_sq
            # This is equivalent to finding insertion point in a list of L_k values.
            # Python's bisect_left(a, x) finds an insertion point for x in a to maintain sorted order.
            # If x is already present in a, the insertion point will be before (to the left of) any existing entries.
            # We need index `k` such that bad_intervals[k-1].L <= next_sq AND bad_intervals[k-1].R >= next_sq
            
            # Search for an interval bad_intervals[k] such that L_k <= next_sq <= R_k
            low = 0
            high = M - 1
            # Find an interval whose L-value is <= next_sq and R-value is >= next_sq
            # This target interval could be any of bad_intervals.
            # More robust: find `k` = index of interval bad_intervals[k] such that bad_intervals[k].L is the largest L_i <= next_sq.
            # Then check if next_sq <= bad_intervals[k].R
            
            # Standard bisect_left to find first interval whose L_i >= next_sq
            # Then check interval before it.
            # bisect_left returns idx: all e in bad_intervals[:idx] have e[0] < next_sq
            # all e in bad_intervals[idx:] have e[0] >= next_sq
            
            # Simplified: check bad_intervals[k] where L_k <= next_sq.
            # The largest k such that L_k <= next_sq.
            search_idx = -1
            b_low, b_high = 0, M - 1
            while b_low <= b_high:
                b_mid = (b_low + b_high) // 2
                if bad_intervals[b_mid][0] <= next_sq:
                    search_idx = b_mid
                    b_low = b_mid + 1
                else:
                    b_high = b_mid - 1
            
            if search_idx != -1 and bad_intervals[search_idx][0] <= next_sq <= bad_intervals[search_idx][1]:
                is_next_sq_bad = True

            if not is_next_sq_bad:
                visited.add(next_sq)
                q.append(next_sq)
                
    print("No")

solve()