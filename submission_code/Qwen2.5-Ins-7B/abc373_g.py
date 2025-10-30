# YOUR CODE HERE
from itertools import permutations

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def do_not_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and orientation(p1, q1, p2) == 0:
        return True
    if o2 == 0 and orientation(p1, q1, q2) == 0:
        return True
    if o3 == 0 and orientation(p2, q2, p1) == 0:
        return True
    if o4 == 0 and orientation(p2, q2, q1) == 0:
        return True
    return False

def find_permutation(N, points, queries):
    for perm in permutations(range(1, N+1)):
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if not do_not_intersect(points[i], queries[perm[i]-1], points[j], queries[perm[j]-1]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return [x+1 for x in perm]
    return [-1]

N = int(input())
points = [list(map(int, input().split())) for _ in range(N)]
queries = [list(map(int, input().split())) for _ in range(N)]
result = find_permutation(N, points, queries)
print(' '.join(map(str, result)))