import collections
import bisect

def solve():
    N, M, A, B = map(int, input().split())

    bad_intervals = []
    for _ in range(M):
        L, R = map(int, input().split())
        bad_intervals.append((L, R))

    # Pre-process L and R values for optimized is_bad check
    L_vals = [interval[0] for interval in bad_intervals]
    R_vals = [interval[1] for interval in bad_intervals]

    def is_bad_optimized(p_val):
        # Find index `idx` such that L_vals[idx-1] <= p_val < L_vals[idx]
        idx = bisect.bisect_right(L_vals, p_val)
        if idx > 0:
            # Check if p_val falls within the previous bad interval
            if R_vals[idx - 1] >= p_val:
                return True
        return False

    # dp[mask] = minimum position `x` for which the window [x, x+B-1] has `mask` as its reachability pattern.
    # mask & (1 << j) == 1 means x+j is reachable.
    dp = collections.defaultdict(lambda: N + 2) # Use N+2 as initial infinity

    # Queue for BFS: stores (mask, current_pos_start_of_window)
    q = collections.deque()

    # Initial state: square 1 is reachable. Window [1, 1+B-1]. Mask (1 << 0) means 1+0 is reachable.
    initial_mask = (1 << 0)
    initial_pos = 1
    
    dp[initial_mask] = initial_pos
    q.append((initial_mask, initial_pos))

    # Precompute critical points to find next_critical_for_window quickly.
    # These are points where the "good/bad" status of squares can change.
    critical_points = set()
    critical_points.add(1) # Start
    critical_points.add(N) # End
    for L, R in bad_intervals:
        critical_points.add(L)
        critical_points.add(R + 1)
    
    # Add points around N. We need to check reachability up to N,
    # and the window can extend past N.
    # Adding N-B to N+B ensures we catch boundaries near N properly.
    for i in range(B + 1):
        if N - i >= 1:
            critical_points.add(N - i)
        critical_points.add(N + i) 
    
    critical_points = sorted(list(critical_points))

    while q:
        curr_mask, curr_pos = q.popleft()

        # If we already found a shorter path to this mask, skip.
        if curr_pos > dp[curr_mask]:
            continue

        # Check if N is reachable within current window [curr_pos, curr_pos + B - 1]
        if N >= curr_pos and N < curr_pos + B:
            if (curr_mask & (1 << (N - curr_pos))):
                print("Yes")
                return
        
        # Calculate the next state (mask and position) if we advance one square linearly.
        next_pos_simulated = curr_pos + 1
        
        # Shift mask to the right (representing current window moving from [x, x+B-1] to [x+1, x+B])
        next_mask_simulated = curr_mask >> 1 
        
        # Check if `curr_pos + B` (the last position in the *next* window `[next_pos_simulated, next_pos_simulated + B - 1]`) 
        # is reachable from `curr_pos + [0, B-1]`.
        # It's reachable if any `curr_pos + k_offset_in_mask` was reachable, where a jump of size `A` to `B` 
        # would land on `curr_pos + B`. This corresponds to `k_offset_in_mask` from `0` to `B-A`.
        can_jump_to_new_last_pos_in_window = False
        for k_offset in range(B - A + 1): # k_offset = 0 corresponds to jump B, k_offset = B-A corresponds to jump A
            if (curr_mask & (1 << k_offset)):
                can_jump_to_new_last_pos_in_window = True
                break
        
        if can_jump_to_new_last_pos_in_window:
            next_mask_simulated |= (1 << (B - 1)) # Set the B-1 bit in the new mask (for next_pos_simulated + (B-1))
        
        # Apply badness and N boundary limits to the `next_mask_simulated`
        # Each bit `k_offset` in `next_mask_simulated` corresponds to `next_pos_simulated + k_offset`.
        for k_offset in range(B): 
            current_square_in_next_window = next_pos_simulated + k_offset
            if current_square_in_next_window > N or is_bad_optimized(current_square_in_next_window):
                next_mask_simulated &= ~(1 << k_offset) # Clear the bit if bad or past N

        # If the mask becomes zero, no path is possible from this point.
        if next_mask_simulated == 0:
            continue

        # Determine if fast-forwarding is possible.
        # Find the next critical point `P` that is strictly after `curr_pos + B - 1`.
        # This `P` marks the end of the current "uniform" segment for the entire window.
        idx_cp = bisect.bisect_right(critical_points, curr_pos + B - 1)
        next_critical_for_window = N + B + 1 # Default, effectively beyond N
        if idx_cp < len(critical_points):
            next_critical_for_window = critical_points[idx_cp]
        
        # Fast-forward condition:
        # 1. `next_pos_simulated` is NOT a new best path for `next_mask_simulated` (i.e., `next_pos_simulated >= dp[next_mask_simulated]`).
        #    This implies we've found a cycle or a longer path to an already known mask state.
        # 2. The current window `[curr_pos, curr_pos + B - 1]` is entirely within a uniform good segment, 
        #    meaning `curr_pos + B - 1` is before `next_critical_for_window`.
        if next_pos_simulated >= dp[next_mask_simulated] and \
           curr_pos + B - 1 < next_critical_for_window:
            
            # Calculate the target position for the start of the window after fast-forwarding.
            # This position will make the window `[target_jump_pos, target_jump_pos + B - 1]` end exactly at `next_critical_for_window`.
            target_jump_pos = next_critical_for_window - (B - 1) 
            
            # Ensure target_jump_pos doesn't go backwards from `next_pos_simulated`, 
            # and it's within sensible bounds relative to N.
            target_jump_pos = max(target_jump_pos, next_pos_simulated)
            target_jump_pos = min(target_jump_pos, N) # Cannot fast-forward window start past N
            
            # If this fast-forwarded position offers a better path for `next_mask_simulated`, add it to queue.
            if target_jump_pos < dp[next_mask_simulated]:
                dp[next_mask_simulated] = target_jump_pos
                q.append((next_mask_simulated, target_jump_pos))
            
        else: # No fast-forwarding applies, or it's a new better path.
              # If `next_pos_simulated` offers a new minimum position for `next_mask_simulated`, update `dp` and queue it.
            if next_pos_simulated < dp[next_mask_simulated]:
                dp[next_mask_simulated] = next_pos_simulated
                q.append((next_mask_simulated, next_pos_simulated))
            
    print("No")