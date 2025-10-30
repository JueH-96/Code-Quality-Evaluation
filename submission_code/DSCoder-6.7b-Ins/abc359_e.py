import sys

def solve():
    N = int(sys.stdin.readline().strip())
    H = list(map(int, sys.stdin.readline().strip().split()))
    A = [0] * (N+1)
    result = [0] * N
    for i in range(N-1, -1, -1):
        if A[i+1] > H[i]:
            A[i] = A[i+1] - (A[i+1] - H[i])
            result[i] = result[i+1] + (A[i+1] - H[i])
        else:
            A[i] = H[i]
            result[i] = result[i+1]
    print(' '.join(map(str, result)))

solve()