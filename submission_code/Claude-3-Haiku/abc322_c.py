import sys

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Initialize the answer list
ans = [0] * N

# Iterate through each day
for i in range(N):
    # Find the first day on or after the current day when fireworks are launched
    for j in range(M):
        if A[j] >= i+1:
            ans[i] = A[j] - i
            break

# Print the answer
for a in ans:
    print(a)