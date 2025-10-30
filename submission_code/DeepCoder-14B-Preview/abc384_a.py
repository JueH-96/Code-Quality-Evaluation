# Read the input values
n, c1, c2 = input().split()
n = int(n)
s = input().strip()

# Process each character in the string
new_s = ''.join([c if c == c1 else c2 for c in s])

# Print the resulting string
print(new_s)