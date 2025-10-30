import sys

def is_good_pair(N, M, A, B):
    # Create a list to store the color of each position
    X = [-1] * (N + 1)

    # Try to color each position with 0 and 1
    for i in range(M):
        if X[A[i]] == -1:
            X[A[i]] = 0
        if X[B[i]] == -1:
            X[B[i]] = 1
        if X[A[i]] == X[B[i]]:
            return False

    return True

# Read inputs
N, M = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))

# Solve the problem
if is_good_pair(N, M, A, B):
    print("Yes")
else:
    print("No")