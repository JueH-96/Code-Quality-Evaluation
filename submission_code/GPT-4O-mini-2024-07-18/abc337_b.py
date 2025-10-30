def is_extended_abc_string(s):
    # Initialize counters for A, B, and C
    count_a = 0
    count_b = 0
    count_c = 0
    
    # State to track which section we are in
    state = 'A'  # Start expecting A's
    
    for char in s:
        if char == 'A':
            if state == 'B' or state == 'C':
                # If we encounter A after B or C, it's invalid
                return "No"
            count_a += 1
        elif char == 'B':
            if state == 'A':
                state = 'B'  # Move to expecting B's
            elif state == 'C':
                # If we encounter B after C, it's invalid
                return "No"
            count_b += 1
        elif char == 'C':
            if state == 'A' or state == 'B':
                state = 'C'  # Move to expecting C's
            count_c += 1
        else:
            # If there's any character other than A, B, or C
            return "No"
    
    # We need at least one of each A, B, and C to form an Extended ABC string
    if count_a > 0 and count_b > 0 and count_c > 0:
        return "Yes"
    else:
        return "No"

# Read input
import sys
input_string = sys.stdin.read().strip()

# Output the result
print(is_extended_abc_string(input_string))