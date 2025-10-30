# Read the number of strings
N = int(input())

# Read N strings
strings = []
for i in range(N):
    strings.append(input())

# Sort strings by length
strings.sort(key=len)

# Concatenate sorted strings
result = ''.join(strings)

# Output the result
print(result)