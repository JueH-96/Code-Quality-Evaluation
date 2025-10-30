# YOUR CODE HERE
import sys

# Read input from stdin
S = sys.stdin.read().strip()

# Check if the string ends with 'san'
if S.endswith('san'):
    print("Yes")
else:
    print("No")