# YOUR CODE HERE

import sys

W, B = map(int, sys.stdin.readline().split())

# The string S is infinitely long and consists of the pattern "wbwbwwbwbwbw" repeated infinitely.
S = "wbwbwwbwbwbw"

# Check if there is a substring of S that consists of W occurrences of 'w' and B occurrences of 'b'.
if W <= len(S) and B <= len(S):
    print("Yes")
else:
    print("No")