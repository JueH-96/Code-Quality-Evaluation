import sys
import bisect

def read_input():
    N, Q = map(int, sys.stdin.readline().split())
    A = sorted(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    K = list(map(int, sys.stdin.readline().split()))
    return N, Q, A, B, K

def solve():
    N, Q, A, B, K = read_input()
    for i in range(Q):
        k = K[i]
        b = B[i]
        idx = bisect.bisect_left(A, b)
        if idx == N and idx > 0:
            print(abs(A[idx-1] - b))
        elif idx == 0:
            print(abs(A[0] - b))
        else:
            print(min(abs(A[idx] - b), abs(A[idx-1] - b)))

solve()