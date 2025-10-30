# YOUR CODE HERE
import sys

# Read input from stdin
A, B = map(int, sys.stdin.read().split())

# Check if A and B are adjacent horizontally
if abs(A - B) == 1:
    print("Yes")
else:
    print("No")