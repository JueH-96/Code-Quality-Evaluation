# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Parse the input
a, b, c, d, e, f = map(int, data[:6])
g, h, i, j, k, l = map(int, data[6:])

# Check for intersection
if (d > g and a < j and e > h and b < k and f > i and c < l):
    print("Yes")
else:
    print("No")