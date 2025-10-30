import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))

    # Create a list to store the result
    res = [0] * N

    # Perform the operations
    for i in range(N):
        res[i] = A[X[i] - 1]

    # If K is greater than 1, repeat the operations
    if K > 1:
        for _ in range(K - 1):
            for i in range(N):
                res[i] = A[res[i] - 1]

    print(' '.join(map(str, res)))

solve()