def can_cross(p1, p2, q1, q2):
    """Check if the line segments p1-q1 and p2-q2 cross each other."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def find_permutation(N, P, Q):
    # Sort P by y-coordinate, and if equal, by x-coordinate
    P_sorted = sorted(P, key=lambda x: (x[1], x[0]))
    
    # Create a list of indices for Q
    indices = list(range(N))
    
    # Try to find a valid permutation
    for perm in itertools.permutations(indices):
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if can_cross(P_sorted[i], Q[perm[i]], P_sorted[j], Q[perm[j]]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return [x + 1 for x in perm]  # Convert to 1-based index
    
    return -1

import itertools
import sys

input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
P = [tuple(map(int, data[i + 1].split())) for i in range(N)]
Q = [tuple(map(int, data[i + 1 + N].split())) for i in range(N)]

result = find_permutation(N, P, Q)

if result == -1:
    print(-1)
else:
    print(" ".join(map(str, result)))