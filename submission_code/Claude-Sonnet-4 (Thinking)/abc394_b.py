# YOUR CODE HERE
n = int(input())
strings = []
for i in range(n):
    strings.append(input())

# Sort by length
strings.sort(key=len)

# Concatenate
result = ''.join(strings)
print(result)