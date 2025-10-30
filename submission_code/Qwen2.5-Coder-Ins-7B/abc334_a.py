# YOUR CODE HERE
import sys

# Read input from stdin
B, G = map(int, sys.stdin.readline().split())

# Compare the costs and print the result
if B > G:
    print("Bat")
else:
    print("Glove")