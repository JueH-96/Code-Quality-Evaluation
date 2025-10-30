# Read input string
S = input().strip()

# Keep only '2' characters and concatenate them
result = ''.join(char for char in S if char == '2')

# Print result
print(result)