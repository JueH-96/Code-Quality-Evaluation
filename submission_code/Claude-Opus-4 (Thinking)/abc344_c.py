# Read N and A
N = int(input())
A = list(map(int, input().split()))

# Read M and B
M = int(input())
B = list(map(int, input().split()))

# Read L and C
L = int(input())
C = list(map(int, input().split()))

# Read Q and X
Q = int(input())
X = list(map(int, input().split()))

# Compute all possible sums of A[i] + B[j]
ab_sums = set()
for a in A:
    for b in B:
        ab_sums.add(a + b)

# For each query, check if X[i] - C[k] is in ab_sums for some k
for x in X:
    if any(x - c in ab_sums for c in C):
        print("Yes")
    else:
        print("No")