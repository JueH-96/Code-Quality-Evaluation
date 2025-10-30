import sys
from itertools import permutations

def ccw(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def do_intersect(p1, p2, q1, q2):
    o1 = ccw(p1, p2, q1)
    o2 = ccw(p1, p2, q2)
    o3 = ccw(q1, q2, p1)
    o4 = ccw(q1, q2, p2)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return False

def solve(N, P, Q):
    for perm in permutations(range(N)):
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if do_intersect(P[i], Q[perm[i]], P[j], Q[perm[j]]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return perm
    return None

N = int(input())
P = [tuple(map(int, input().split())) for _ in range(N)]
Q = [tuple(map(int, input().split())) for _ in range(N)]

result = solve(N, P, Q)
if result is None:
    print(-1)
else:
    print(' '.join(map(lambda x: str(x + 1), result)))