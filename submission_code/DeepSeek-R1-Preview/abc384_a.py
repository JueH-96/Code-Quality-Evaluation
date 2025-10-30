# Read the input values
n, c1, c2 = input().split()
s = input().strip()

# Process each character in the string
result = ''.join([c1 if ch == c1 else c2 for ch in s])

# Print the result
print(result)