# Read N and S from input
N = int(input())
S = input().strip()

# Validate input length
assert len(S) == N, "Length of S does not match N"

# Create the new string by concatenating each character twice
result = ''.join([char*2 for char in S])

# Print the result
print(result)