import sys

# Read the input string from standard input and remove leading/trailing whitespace
S = sys.stdin.readline().strip()

# Initialize the state variable.
# 0: Expecting A's (or transition to B or C)
# 1: Expecting B's (or transition to C, but no more A's allowed)
# 2: Expecting C's (no more A's or B's allowed)
current_state = 0 

# Flag to track if the string remains valid throughout the check
is_valid_extended_abc = True

# Iterate through each character of the input string S
for char in S:
    if current_state == 0:
        # Currently in the potential 'A' section, or initial state
        if char == 'A':
            # Character is 'A', stay in 'A' section
            pass 
        elif char == 'B':
            # Character is 'B', transition to 'B' section
            current_state = 1
        elif char == 'C':
            # Character is 'C', transition to 'C' section (skipped 'A' and 'B' sections)
            current_state = 2
    elif current_state == 1:
        # Currently in the potential 'B' section (implies 'A' section has passed)
        if char == 'A':
            # Invalid: 'A' appeared after 'B' (e.g., AB...A)
            is_valid_extended_abc = False
            break # No need to check further, it's invalid
        elif char == 'B':
            # Character is 'B', stay in 'B' section
            pass
        elif char == 'C':
            # Character is 'C', transition to 'C' section
            current_state = 2
    elif current_state == 2:
        # Currently in the potential 'C' section (implies 'A' and 'B' sections have passed)
        if char == 'A' or char == 'B':
            # Invalid: 'A' or 'B' appeared after 'C' (e.g., BC...A or AC...B)
            is_valid_extended_abc = False
            break # No need to check further, it's invalid
        elif char == 'C':
            # Character is 'C', stay in 'C' section
            pass

# After checking all characters, print the result
if is_valid_extended_abc:
    sys.stdout.write("Yes
")
else:
    sys.stdout.write("No
")