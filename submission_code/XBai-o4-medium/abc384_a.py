# Read input values
n, c1, c2 = input().split()
n = int(n)
s = input().strip()

# Generate the result string
result = ''.join([char if char == c1 else c2 for char in s])

# Print the result
print(result)