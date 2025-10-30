import sys

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve the problem for each K
B = []
for K in range(1, N+1):
    # Initialize Takahashi's size to A[K-1]
    size = A[K-1]
    
    # Absorb adjacent smaller slimes
    for i in range(K-2, -1, -1):
        if A[i] < size:
            size += A[i]
        else:
            break
    for i in range(K, N):
        if A[i] < size:
            size += A[i]
        else:
            break
    
    B.append(size)

# Print the output
print(" ".join(map(str, B)))