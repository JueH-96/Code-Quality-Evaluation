# Read input values
n, c1, c2 = input().split()
n = int(n)
S = input().strip()

# Process each character in S
result = []
for char in S:
    if char != c1:
        result.append(c2)
    else:
        result.append(char)

# Join and print the result
print(''.join(result))