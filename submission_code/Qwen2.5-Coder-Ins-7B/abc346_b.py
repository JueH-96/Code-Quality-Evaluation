# YOUR CODE HERE
import sys

# Read input from stdin
W, B = map(int, sys.stdin.read().split())

# The repeating pattern is 'wbwbwwbwbwbw'
pattern = 'wbwbwwbwbwbw'

# Check if we can form a substring with W 'w's and B 'b's
def can_form_substring(W, B):
    # The pattern has 12 characters
    pattern_length = len(pattern)
    # The maximum number of 'w's and 'b's in any substring of the pattern
    max_w = pattern.count('w')
    max_b = pattern.count('b')
    
    # Check if we can form the required number of 'w's and 'b's
    if W <= max_w and B <= max_b:
        return True
    return False

# Output the result
if can_form_substring(W, B):
    print("Yes")
else:
    print("No")