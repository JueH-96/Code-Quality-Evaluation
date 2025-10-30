# YOUR CODE HERE
import sys

# Function to check if 'sub' is a subsequence of 's'
# 's' is lowercase, 'sub' is uppercase. Comparison is case-insensitive.
def is_subsequence(s, sub):
    i = 0 # Pointer for string s
    j = 0 # Pointer for string sub
    while i < len(s) and j < len(sub):
        # Convert s[i] to uppercase for comparison with sub[j]
        if s[i].upper() == sub[j]:
            j += 1 # Found a character for the subsequence, move to the next in sub
        i += 1 # Always move to the next character in s
    
    # If we found all characters of sub, j will equal len(sub)
    return j == len(sub)

# Read input strings
S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

# Check if T is an airport code for S based on the rules

# Rule 1: T is a 3-character subsequence of S (converted to uppercase)
if is_subsequence(S, T):
    print("Yes")
else:
    # Rule 2: T is a 2-character subsequence of S (converted to uppercase) + 'X'
    # This rule applies only if T's third character is 'X'.
    # If Rule 1 failed, check Rule 2.
    if T[2] == 'X' and is_subsequence(S, T[:2]):
        print("Yes")
    else:
        # Neither rule is satisfied
        print("No")