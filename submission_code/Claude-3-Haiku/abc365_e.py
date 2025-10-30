import sys

# Read input
N = int(input())
A = [int(x) for x in input().split()]

# Compute the answer
total = 0
for i in range(N-1):
    xor_val = 0
    for j in range(i, N):
        xor_val ^= A[j]
        total += xor_val

print(total)