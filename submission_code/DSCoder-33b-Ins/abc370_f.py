# YOUR CODE HERE

import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A.append(A[0])

    total = sum(A)
    min_mass = total // K
    if total % K != 0:
        min_mass += 1

    mass = 0
    cuts = 0
    for i in range(N):
        mass += A[i]
        if mass >= min_mass:
            cuts += 1
            mass = 0

    print(min_mass, cuts)

solve()