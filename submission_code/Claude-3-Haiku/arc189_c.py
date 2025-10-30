import sys

def solve():
    N, X = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    P = list(map(int, input().split()))
    Q = list(map(int, input().split()))

    # Check if all boxes other than the X-th box are already empty
    if all(a == 0 and b == 0 for a, b in zip(A[:X-1], B[:X-1])) and all(a == 0 and b == 0 for a, b in zip(A[X:], B[X:])):
        return 0

    # Perform the operations
    operations = 0
    for i in range(1, N+1):
        if A[i-1] + B[i-1] > 0:
            A[P[i-1]-1] += A[i-1]
            B[Q[i-1]-1] += B[i-1]
            A[i-1] = 0
            B[i-1] = 0
            operations += 1

            # Check if all boxes other than the X-th box are empty
            if all(a == 0 and b == 0 for a, b in zip(A[:X-1], B[:X-1])) and all(a == 0 and b == 0 for a, b in zip(A[X:], B[X:])):
                return operations

    # If it's not possible to achieve the goal, return -1
    return -1

print(solve())