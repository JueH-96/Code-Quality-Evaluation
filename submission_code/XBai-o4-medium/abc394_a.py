# Read the input string
s = input().strip()

# Filter out all characters except '2' and join them
result = ''.join([c for c in s if c == '2'])

# Print the result
print(result)