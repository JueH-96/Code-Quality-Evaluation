# Read the first line and split into parts
parts = input().split()
n = int(parts[0])
c1 = parts[1]
c2 = parts[2]

# Read the string S
s = input().strip()

# Replace each character not equal to c1 with c2
result = ''.join([c if c == c1 else c2 for c in s])

# Print the result
print(result)