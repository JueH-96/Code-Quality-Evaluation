import sys
import bisect
from collections import defaultdict

# Set higher recursion depth for deep structures if needed, though not directly used by this iterative DP.
# sys.setrecursionlimit(200000) 

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    
    ops = []
    for i in range(Q):
        # Read P_i, V_i for the i-th operation
        ops.append(list(map(int, sys.stdin.readline().split()))) 

    MOD = 998244353

    # Coordinate Compression
    # Collect all relevant coordinates: 1, N+1, and for each operation i, P_i and P_i+1 (if P_i < N).
    # These points define the boundaries of elementary intervals where the sequence S value is constant.
    coords = {1} 
    # N+1 serves as the coordinate after N.
    coords.add(N + 1) 
    
    for p, v in ops:
        coords.add(p)
        # P_i+1 is relevant because range [P_i, N] starts at P_i, and range [1, P_i] ends at P_i.
        # The interval containing P_i is important. Its behavior might differ from adjacent intervals.
        # Including P_i+1 ensures intervals like [P_i, P_i] are captured if P_i+1 is different from P_i.
        if p + 1 <= N:
             coords.add(p + 1)

    # Sort the unique coordinates to define the elementary intervals.
    sorted_coords = sorted(list(coords))
    
    # M is the number of elementary intervals.
    # An interval k corresponds to the range [sorted_coords[k], sorted_coords[k+1]-1] in the original sequence S.
    # The segment tree (or tuple state) will have M leaves/elements, indexed 0 to M-1.
    M = len(sorted_coords) - 1 

    # Initial state: The sequence S is all zeros. This corresponds to a tuple of M zeros.
    # dp maps state (represented by tuple) to the number of ways to reach that state.
    initial_state = tuple([0] * M)
    dp = {initial_state: 1} # Start with 1 way to reach the initial state.

    # Process operations one by one
    for i in range(Q):
        Pi, Vi = ops[i]
        
        # If dp is empty, it means no valid sequence of choices could reach this step.
        # The total count will be 0. We can terminate early.
        if not dp:
             print(0)
             return

        # Use defaultdict for accumulating counts for the next states.
        new_dp = defaultdict(int)

        # Find the leaf index k corresponding to the elementary interval containing Pi.
        # Interval k spans range [sorted_coords[k], sorted_coords[k+1]-1].
        # We need k such that sorted_coords[k] <= Pi < sorted_coords[k+1].
        # bisect_right finds the insertion point `idx` for Pi. All elements to the left of `idx` are <= Pi.
        # The coordinate Pi is contained within the interval starting at sorted_coords[idx-1].
        # The leaf index corresponding to this interval is `idx-1`.
        k_upper = bisect.bisect_right(sorted_coords, Pi) 
        target_leaf_idx = k_upper - 1 # 0-based leaf index

        # Determine the ranges of leaf indices affected by the two choices.
        # Choice L affects S_1..S_{Pi}. This corresponds to leaves 0 through target_leaf_idx.
        range_L_indices = (0, target_leaf_idx) # inclusive range [0, target_leaf_idx]
        
        # Choice R affects S_{Pi}..S_N. This corresponds to leaves target_leaf_idx through M-1.
        range_R_indices = (target_leaf_idx, M - 1) # inclusive range [target_leaf_idx, M-1]
        
        # Store dp.items() to avoid issues if dict changes size during iteration (though not strictly necessary here).
        current_dp_items = list(dp.items()) 

        for current_state_tuple, count in current_dp_items:
            
            # --- Try Choice L: Replace S_1..S_{Pi} with V_i ---
            max_val_L = 0
            # Find the maximum value in the affected range for choice L. O(M) complexity part.
            for k in range(range_L_indices[0], range_L_indices[1] + 1):
                 max_val_L = max(max_val_L, current_state_tuple[k])

            # Check validity condition: Current max must be <= V_i.
            if max_val_L <= Vi:
                # Create the next state by updating the tuple elements. Convert to list for modification.
                new_state_list = list(current_state_tuple)
                # Update values in the affected range. O(M) complexity part.
                for k in range(range_L_indices[0], range_L_indices[1] + 1):
                    new_state_list[k] = Vi
                
                # Convert back to tuple to use as dictionary key.
                new_state_tuple_L = tuple(new_state_list)
                # Add the count to the new state entry in new_dp, modulo MOD.
                new_dp[new_state_tuple_L] = (new_dp[new_state_tuple_L] + count) % MOD

            # --- Try Choice R: Replace S_{Pi}..S_N with V_i ---
            max_val_R = 0
            # Find the maximum value in the affected range for choice R. O(M) complexity part.
            for k in range(range_R_indices[0], range_R_indices[1] + 1):
                max_val_R = max(max_val_R, current_state_tuple[k])

            # Check validity condition: Current max must be <= V_i.
            if max_val_R <= Vi:
                # Create the next state. Convert tuple to list.
                new_state_list = list(current_state_tuple)
                # Update values in the affected range. O(M) complexity part.
                for k in range(range_R_indices[0], range_R_indices[1] + 1):
                     new_state_list[k] = Vi
                
                # Convert back to tuple.
                new_state_tuple_R = tuple(new_state_list)
                # Add the count to the new state entry, modulo MOD.
                # defaultdict handles the case where this state might have already been added (e.g., by Choice L).
                new_dp[new_state_tuple_R] = (new_dp[new_state_tuple_R] + count) % MOD
        
        # Update dp for the next iteration.
        dp = new_dp
       
    # After processing all Q operations, the total count is the sum of counts for all reachable final states.
    total_count = 0
    for count in dp.values():
        total_count = (total_count + count) % MOD
    
    print(total_count)

# Call the main solver function.
solve()