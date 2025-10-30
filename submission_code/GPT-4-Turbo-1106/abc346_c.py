# Read the input values
N, K = map(int, input().split())
A = set(map(int, input().split()))

# Calculate the sum of integers between 1 and K that are not in A
missing_sum = sum(i for i in range(1, K + 1) if i not in A)

# Print the result
print(missing_sum)