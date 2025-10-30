# Read the input values
n, c1, c2 = input().split()
n = int(n)
s = input().strip()

# Process each character in the string
result = [c if c == c1 else c2 for c in s]

# Output the result
print(''.join(result))