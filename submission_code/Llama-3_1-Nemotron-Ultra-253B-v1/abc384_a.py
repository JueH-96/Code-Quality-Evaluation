# Read input values
n, c1, c2 = input().split()
n = int(n)
s = input().strip()

# Replace each character not equal to c1 with c2
result = ''.join([char if char == c1 else c2 for char in s])

# Output the result
print(result)