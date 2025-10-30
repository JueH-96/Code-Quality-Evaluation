# YOUR CODE HERE
import sys

# Read the input string
S = sys.stdin.read().strip()

# Check if the sorted version of the string is "ABC"
if sorted(S) == list("ABC"):
    print("Yes")
else:
    print("No")