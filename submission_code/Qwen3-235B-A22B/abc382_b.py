# Read input
N, D = map(int, input().split())
S = input()

# Collect positions of all '@' characters
positions = [i for i, char in enumerate(S) if char == '@']

# Calculate how many cookies remain
remaining = len(positions) - D

# Initialize the result with all '.'s
result = ['.'] * N

# Mark the positions of the remaining cookies
for i in range(remaining):
    idx = positions[i]
    result[idx] = '@'

# Print the result
print(''.join(result))