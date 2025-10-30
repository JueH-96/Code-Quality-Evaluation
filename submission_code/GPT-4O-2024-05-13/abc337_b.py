# YOUR CODE HERE
def is_extended_abc_string(s):
    # Check if the string can be split into three parts: A's, B's, and C's
    i = 0
    n = len(s)
    
    # Count A's
    while i < n and s[i] == 'A':
        i += 1
    # Count B's
    while i < n and s[i] == 'B':
        i += 1
    # Count C's
    while i < n and s[i] == 'C':
        i += 1
    
    # If we have consumed the entire string, it is a valid Extended ABC string
    return i == n

# Read input
import sys
input = sys.stdin.read().strip()

# Check and print result
if is_extended_abc_string(input):
    print("Yes")
else:
    print("No")