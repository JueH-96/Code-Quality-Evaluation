# YOUR CODE HERE
def can_form_substring(W, B):
    # The repeating pattern
    pattern = "wbwbwwbwbwbw"
    pattern_length = len(pattern)
    
    # Count of 'w' and 'b' in one cycle
    count_w_in_cycle = pattern.count('w')
    count_b_in_cycle = pattern.count('b')
    
    # If W or B is larger than what a single cycle can provide, we need multiple cycles
    if W > count_w_in_cycle or B > count_b_in_cycle:
        # Calculate full cycles needed
        full_cycles_w = (W + count_w_in_cycle - 1) // count_w_in_cycle
        full_cycles_b = (B + count_b_in_cycle - 1) // count_b_in_cycle
        
        # We need at least max(full_cycles_w, full_cycles_b) full cycles
        full_cycles_needed = max(full_cycles_w, full_cycles_b)
        
        # Calculate remaining w and b after full cycles
        remaining_w = W - full_cycles_needed * count_w_in_cycle
        remaining_b = B - full_cycles_needed * count_b_in_cycle
        
        # Check if the remaining can be satisfied within one more cycle
        if remaining_w <= 0 and remaining_b <= 0:
            return "Yes"
        else:
            # Check if a partial cycle can satisfy the remaining
            for start in range(pattern_length):
                for length in range(1, pattern_length + 1):
                    substring = pattern[start:start + length]
                    if len(substring) > pattern_length:
                        break
                    if substring.count('w') == remaining_w and substring.count('b') == remaining_b:
                        return "Yes"
            return "No"
    else:
        # Check within one cycle
        for start in range(pattern_length):
            for length in range(1, pattern_length + 1):
                substring = pattern[start:start + length]
                if len(substring) > pattern_length:
                    break
                if substring.count('w') == W and substring.count('b') == B:
                    return "Yes"
        return "No"

import sys
input = sys.stdin.read
W, B = map(int, input().strip().split())
print(can_form_substring(W, B))