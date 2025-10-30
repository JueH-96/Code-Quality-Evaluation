# YOUR CODE HERE

import sys

S = sys.stdin.readline().strip()

if S[:3] == "ABC" and S[3:].isdigit() and 1 <= int(S[3:]) <= 348:
    print("Yes")
else:
    print("No")