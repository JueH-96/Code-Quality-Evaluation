import math

# Read the number of elements in the sequence
N = int(input().strip())

# Read the sequence of integers
A = list(map(int, input().strip().split()))

# Initialize the sum
total_sum = 0

# Calculate the sum of the floor of ratios
for i in range(N - 1):
    for j in range(i + 1, N):
        total_sum += math.floor(max(A[i], A[j]) / min(A[i], A[j]))

# Print the result
print(total_sum)