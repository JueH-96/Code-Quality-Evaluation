# Read input
n, c1, c2 = input().split()
n = int(n)
s = input().strip()

# Process each character
result = [c2 if ch != c1 else ch for ch in s]

# Output the result
print(''.join(result))