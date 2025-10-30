# Read input
n, c1, c2 = input().split()
n = int(n)
s = input().strip()

# Process each character in the string
result = []
for char in s:
    if char == c1:
        result.append(char)
    else:
        result.append(c2)

# Output the result
print(''.join(result))