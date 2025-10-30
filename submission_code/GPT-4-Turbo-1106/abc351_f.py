# Read the number of elements in the sequence
N = int(input().strip())

# Read the sequence of integers
A = list(map(int, input().strip().split()))

# Initialize the sum
total_sum = 0

# Calculate the expression
for i in range(N):
    for j in range(i + 1, N):
        total_sum += max(A[j] - A[i], 0)

# Print the result
print(total_sum)