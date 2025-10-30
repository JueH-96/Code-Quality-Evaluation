# Read the input string
S = input().strip()

# Filter out all characters except '2's and join them
result = ''.join([c for c in S if c == '2'])

# Print the resulting string
print(result)