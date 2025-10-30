# YOUR CODE HERE
import sys

def read_input():
    N = int(input())
    P = [tuple(map(int, input().split())) for _ in range(N)]
    Q = [tuple(map(int, input().split())) for _ in range(N)]
    return N, P, Q

def solve(N, P, Q):
    # Sort Q by the second coordinate
    Q.sort(key=lambda x: x[1])
    # Create a list of indices
    indices = list(range(N))
    # Sort indices by the first coordinate of P
    indices.sort(key=lambda i: P[i])
    # Create a list of R
    R = [0] * N
    for i in range(N):
        R[indices[i]] = Q[i][0]
    return R

def check(N, P, Q, R):
    # Check if any two line segments intersect
    for i in range(N):
        for j in range(i+1, N):
            # Check if P_i and P_j intersect with Q_R_i and Q_R_j
            if (P[i][1] - P[j][1]) * (Q[R[i]][0] - Q[R[j]][0]) == (P[i][0] - P[j][0]) * (Q[R[i]][1] - Q[R[j]][1]):
                return False
    return True

def main():
    N, P, Q = read_input()
    R = solve(N, P, Q)
    if check(N, P, Q, R):
        print(' '.join(map(str, R)))
    else:
        print(-1)

if __name__ == '__main__':
    main()