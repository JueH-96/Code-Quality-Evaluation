# YOUR CODE HERE
def can_rearrange_to_abc(s):
    # Check if the sorted version of the input string matches "ABC"
    return sorted(s) == ['A', 'B', 'C']

# Read input from standard input
import sys
input = sys.stdin.read().strip()

# Determine if it can be rearranged to "ABC"
if can_rearrange_to_abc(input):
    print("Yes")
else:
    print("No")