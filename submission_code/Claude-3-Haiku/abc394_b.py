# Read the input
N = int(input())
strings = []
for _ in range(N):
    strings.append(input())

# Sort the strings by length
strings.sort(key=len)

# Concatenate the sorted strings
result = ''.join(strings)

# Print the result
print(result)