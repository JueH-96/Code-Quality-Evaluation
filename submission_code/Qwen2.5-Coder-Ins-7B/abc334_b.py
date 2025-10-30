# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

A = int(data[0])
M = int(data[1])
L = int(data[2])
R = int(data[3])

# Find the smallest k such that A + kM >= L
k1 = (L - A + M - 1) // M
# Find the largest k such that A + kM <= R
k2 = (R - A) // M

# The number of Christmas trees between L and R (inclusive) is k2 - k1 + 1
result = k2 - k1 + 1

print(result)