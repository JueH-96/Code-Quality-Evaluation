import sys

def solve():
    N = int(sys.stdin.readline())
    A_str = sys.stdin.readline().strip()

    # Calculate val_orig: the final single bit if no changes are made to A
    # current_A_chars stores the characters of the string at the current level of operation
    current_A_chars = list(A_str) 
    
    # This loop simulates the operation N times.
    # _op_level_idx ranges from 0 to N-1.
    # After k applications of the operation, the string length is 3^(N-k).
    # So after N applications, length is 3^0 = 1.
    for _op_level_idx in range(N):
        next_A_chars = []
        current_len = len(current_A_chars)
        num_groups = current_len // 3
        
        for i in range(num_groups):
            # Extract the i-th group of 3 characters
            group = current_A_chars[3*i : 3*i+3]
            
            # Calculate sum of bits in the group
            s = 0
            for char_val in group:
                s += int(char_val) # '0' becomes 0, '1' becomes 1
            
            # Determine majority: '1' if sum is 2 or 3, '0' if sum is 0 or 1
            if s >= 2: 
                next_A_chars.append('1')
            else: 
                next_A_chars.append('0')
        
        current_A_chars = next_A_chars
        # This check is for robustness, but problem constraints should prevent empty list
        if not current_A_chars: 
            # This should not be reached if N is valid for A_str length (3^N)
            break 
            
    val_orig = int(current_A_chars[0])

    # Dynamic Programming calculation
    # dp_costs[j] stores a tuple (cost_to_make_bit_j_zero, cost_to_make_bit_j_one)
    # Initially, dp_costs corresponds to the costs for individual bits of A (level 0)
    dp_costs = []
    for char_a in A_str:
        if char_a == '0':
            dp_costs.append((0, 1)) # (cost for target 0, cost for target 1)
        else: # char_a == '1'
            dp_costs.append((1, 0))

    # This loop iterates N times, reducing the length of dp_costs by a factor of 3 each time.
    # It computes costs for level 1, then level 2, ..., up to level N.
    # After N iterations, dp_costs will contain one tuple for the final single bit.
    for _dp_level_idx in range(N):
        next_dp_costs = []
        current_len = len(dp_costs)
        num_groups = current_len // 3

        for i in range(num_groups):
            # Costs for the three "children" bits from the previous level
            # cX0 is min cost for child X (0, 1, or 2) to be 0
            # cX1 is min cost for child X (0, 1, or 2) to be 1
            (c00, c01) = dp_costs[3*i]
            (c10, c11) = dp_costs[3*i+1]
            (c20, c21) = dp_costs[3*i+2]

            # Calculate minimum cost to make the current composite bit 0.
            # This requires the majority of its three children (v0,v1,v2) to be 0.
            # Possible (v0,v1,v2) combinations: (0,0,0), (0,0,1), (0,1,0), (1,0,0)
            cost_for_0_val = min(
                c00 + c10 + c20,  # Children (0,0,0)
                c00 + c10 + c21,  # Children (0,0,1)
                c00 + c11 + c20,  # Children (0,1,0)
                c01 + c10 + c20   # Children (1,0,0)
            )

            # Calculate minimum cost to make the current composite bit 1.
            # This requires the majority of its three children (v0,v1,v2) to be 1.
            # Possible (v0,v1,v2) combinations: (1,1,1), (1,1,0), (1,0,1), (0,1,1)
            cost_for_1_val = min(
                c01 + c11 + c21,  # Children (1,1,1)
                c01 + c11 + c20,  # Children (1,1,0)
                c01 + c10 + c21,  # Children (1,0,1)
                c00 + c11 + c21   # Children (0,1,1)
            )
            next_dp_costs.append((cost_for_0_val, cost_for_1_val))
        
        dp_costs = next_dp_costs
        # This check is for robustness
        if not dp_costs:
            # This should not be reached if N is valid
            break

    # After N levels of DP, dp_costs contains a single tuple:
    # (min_cost_for_final_bit_to_be_0, min_cost_for_final_bit_to_be_1)
    final_tuple_costs = dp_costs[0] 
    
    # We want to flip val_orig.
    # If val_orig is 0, we want it to be 1. Cost is final_tuple_costs[1].
    # If val_orig is 1, we want it to be 0. Cost is final_tuple_costs[0].
    if val_orig == 0:
        print(final_tuple_costs[1]) 
    else: # val_orig == 1
        print(final_tuple_costs[0])

solve()