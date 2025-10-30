# Read input
N = int(input())
S = input()

# Initialize result
result = 0

# For each starting position i
for i in range(N):
    # For each ending position j >= i
    for j in range(i, N):
        # Extract substring and convert to integer
        num = int(S[i:j+1])
        # Add to result
        result += num

# Print result
print(result)