# YOUR CODE HERE
import sys

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def find_permutation(N, P, Q):
    from itertools import permutations
    for R in permutations(range(N)):
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if intersect(P[i], Q[R[i]], P[j], Q[R[j]]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return [r+1 for r in R]
    return -1

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    P = [(int(input[2*i+1]), int(input[2*i+2])) for i in range(N)]
    Q = [(int(input[2*N+2*i+1]), int(input[2*N+2*i+2])) for i in range(N)]
    result = find_permutation(N, P, Q)
    if result == -1:
        print(-1)
    else:
        print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()