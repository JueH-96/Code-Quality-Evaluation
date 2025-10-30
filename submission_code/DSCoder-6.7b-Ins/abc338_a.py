# YOUR CODE HERE

import sys

S = sys.stdin.readline().strip()

if S[0].isupper():
    if S[1:].islower():
        print("Yes")
    else:
        print("No")
else:
    print("No")