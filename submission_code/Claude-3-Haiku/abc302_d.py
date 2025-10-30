import sys

# Read input
N, M, D = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort the lists
A.sort()
B.sort()

# Initialize variables
max_sum = -1
i, j = 0, 0

# Iterate through the lists
while i < N and j < M:
    diff = abs(A[i] - B[j])
    if diff <= D:
        max_sum = max(max_sum, A[i] + B[j])
        i += 1
        j += 1
    elif A[i] < B[j]:
        i += 1
    else:
        j += 1

# Print the result
print(max_sum)