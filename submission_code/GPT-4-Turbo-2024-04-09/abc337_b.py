def is_extended_abc_string(s):
    # State 0: Expecting 'A'
    # State 1: Expecting 'B'
    # State 2: Expecting 'C'
    state = 0
    
    for char in s:
        if state == 0:
            if char == 'A':
                continue
            elif char == 'B':
                state = 1
            else:
                return "No"
        elif state == 1:
            if char == 'B':
                continue
            elif char == 'C':
                state = 2
            else:
                return "No"
        elif state == 2:
            if char == 'C':
                continue
            else:
                return "No"
    
    return "Yes"

# Read input
import sys
input = sys.stdin.read
s = input().strip()

# Output the result
print(is_extended_abc_string(s))