import math

def solve():
    N = int(input())
    D_list = list(map(int, input().split()))
    L1, C1, K1_limit = map(int, input().split())
    L2, C2, K2_limit = map(int, input().split())

    # dp[s1_count] stores the minimum number of type-2 sensors required.
    # s1_count is the total number of type-1 sensors used.
    
    infinity_val = float('inf') 
    
    # Initialize dp table for 0 sections processed.
    # dp[0] = 0: using 0 type-1 sensors requires 0 type-2 sensors.
    # Other states are initially unreachable (infinite type-2 sensors).
    dp = [infinity_val] * (K1_limit + 1)
    dp[0] = 0

    # Process each section
    for i in range(N):
        D_current = D_list[i]
        dp_next = [infinity_val] * (K1_limit + 1)

        # Calculate X_max_for_D_current: max type-1 sensors useful for D_current.
        # Using more type-1 sensors for D_current gives no further reduction in type-2 sensors (already 0).
        # If D_current is 0, X_max_for_D_current is 0. Otherwise, ceil(D_current / L1).
        # This calculation works even for D_current = 0: (0 + L1 - 1) // L1 is 0 if L1 >= 1.
        X_max_for_D_current = (D_current + L1 - 1) // L1 if D_current > 0 else 0
        
        # Iterate over s1_total_prev: number of type-1 sensors used for sections before current one.
        for s1_total_prev in range(K1_limit + 1):
            if dp[s1_total_prev] == infinity_val:
                continue # This state was unreachable for previous sections.

            # Iterate over x1_current: number of type-1 sensors to use for D_current.
            # x1_current ranges from 0 up to X_max_for_D_current.
            for x1_current in range(X_max_for_D_current + 1):
                s1_total_curr = s1_total_prev + x1_current
                
                if s1_total_curr > K1_limit:
                    # Total type-1 sensors would exceed K1_limit.
                    # Since x1_current increases, further iterations for this s1_total_prev are also invalid.
                    break 
                
                # Calculate type-2 sensors needed for D_current with x1_current type-1 sensors.
                remaining_D_to_cover_by_L2 = D_current - x1_current * L1
                y2_current = 0 # Number of type-2 sensors for this section.
                if remaining_D_to_cover_by_L2 > 0:
                    y2_current = (remaining_D_to_cover_by_L2 + L2 - 1) // L2
                
                # Check if this specific configuration (x1_current, y2_current) is valid locally.
                if y2_current > K2_limit: 
                    # This choice of x1_current requires too many type-2 sensors for D_current alone.
                    continue 
                
                # Total type-2 sensors if we use this configuration for D_current.
                current_s2_sum = dp[s1_total_prev] + y2_current
                
                # Update dp_next if this path is better (fewer type-2 sensors for s1_total_curr type-1 sensors).
                if current_s2_sum < dp_next[s1_total_curr]:
                    dp_next[s1_total_curr] = current_s2_sum
        
        # Move to the next state.
        dp = dp_next

    min_total_cost = infinity_val
    # After processing all N sections, find the minimum cost.
    for s1_final_count in range(K1_limit + 1):
        if dp[s1_final_count] == infinity_val:
            continue # This number of type-1 sensors is not achievable.
        
        s2_final_count = dp[s1_final_count]
        
        # Final check on total type-2 sensors limit.
        if s2_final_count > K2_limit:
            continue # Exceeds K2_limit for type-2 sensors.
            
        cost = s1_final_count * C1 + s2_final_count * C2
        if cost < min_total_cost:
            min_total_cost = cost
            
    if min_total_cost == infinity_val:
        print("-1")
    else:
        print(min_total_cost)

solve()