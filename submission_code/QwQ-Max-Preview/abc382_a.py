# Read input
N, D = map(int, input().split())
S = input().strip()

# Count the number of '@' characters
C = S.count('@')

# Calculate the result
result = (N - C) + D

# Output the answer
print(result)