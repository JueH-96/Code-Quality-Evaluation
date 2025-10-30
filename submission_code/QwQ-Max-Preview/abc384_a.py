# Read input
n, c1, c2 = input().split()
s = input().strip()

# Process each character in the string
result = []
for char in s:
    if char == c1:
        result.append(c1)
    else:
        result.append(c2)

# Output the result
print(''.join(result))