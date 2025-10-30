# Read input values
N, M, C, K = map(int, input().split())
A = list(map(int, input().split()))

# Initialize the sum
total_sum = 0

# Calculate the sum of minimum values for each k
for k in range(K):
    min_value = float('inf')
    for i in range(N):
        min_value = min(min_value, (C * k + A[i]) % M)
    total_sum += min_value

# Print the result
print(total_sum)