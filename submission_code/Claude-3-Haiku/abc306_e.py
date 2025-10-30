import sys

# Read input
N, K, Q = map(int, input().split())
updates = []
for _ in range(Q):
    X, Y = map(int, input().split())
    updates.append((X, Y))

# Initialize sequence A
A = [0] * N

# Apply updates and compute f(A) after each update
for X, Y in updates:
    A[X-1] = Y
    B = sorted(A, reverse=True)
    print(sum(B[:K]))