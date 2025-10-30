# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

if len(set(A)) == 1:
    print("Yes")
else:
    print("No")