# YOUR CODE HERE
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(P1, Q1, P2, Q2):
    return ccw(P1, P2, Q2) != ccw(Q1, P2, Q2) and ccw(P1, Q1, P2) != ccw(P1, Q1, Q2)

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]
Q = [(int(data[2*N+2*i+1]), int(data[2*N+2*i+2])) for i in range(N)]

from itertools import permutations

for perm in permutations(range(N)):
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if intersect(P[i], Q[perm[i]], P[j], Q[perm[j]]):
                valid = False
                break
        if not valid:
            break
    if valid:
        print(' '.join(str(x+1) for x in perm))
        break
else:
    print(-1)