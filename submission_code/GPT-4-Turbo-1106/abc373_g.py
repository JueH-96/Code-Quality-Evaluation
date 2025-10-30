import sys
from itertools import permutations

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def find_permutation(N, P, Q):
    for perm in permutations(range(N)):
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if intersect(P[i], Q[perm[i]], P[j], Q[perm[j]]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return [x + 1 for x in perm]
    return None

def main():
    N = int(sys.stdin.readline().strip())
    P = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    Q = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]

    R = find_permutation(N, P, Q)
    if R is None:
        print(-1)
    else:
        print(' '.join(map(str, R)))

if __name__ == "__main__":
    main()