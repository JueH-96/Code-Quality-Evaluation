# Read input
N, D = map(int, input().split())
S = input().strip()

# Calculate the number of @ in S
count_at = S.count('@')

# Compute the result
result = (N - count_at) + D

# Output the result
print(result)