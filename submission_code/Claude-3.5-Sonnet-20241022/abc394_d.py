def is_colorful_bracket_sequence(s):
    # Keep trying to remove valid bracket pairs until no more changes can be made
    while True:
        # Try to find any valid bracket pair
        found = False
        n = len(s)
        
        # Check each position for possible bracket pairs
        for i in range(n-1):
            if (s[i:i+2] == '()' or 
                s[i:i+2] == '[]' or 
                s[i:i+2] == '<>'):
                # Remove the bracket pair and concatenate remaining parts
                s = s[:i] + s[i+2:]
                found = True
                break
        
        # If no valid pairs found and string is not empty, it's not a valid sequence
        if not found:
            return len(s) == 0

# Read input
S = input().strip()

# Process and output result
if is_colorful_bracket_sequence(S):
    print("Yes")
else:
    print("No")