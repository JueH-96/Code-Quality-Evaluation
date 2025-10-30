# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

count = 0
while len(set(A)) > 1:
    A.sort(reverse=True)
    A[0] -= 1
    A[1] -= 1
    count += 1
    A = [i for i in A if i > 0]

print(count)