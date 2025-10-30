# YOUR CODE HERE
n = int(input())
strings = []
for _ in range(n):
    strings.append(input().strip())

# Sort strings by length in ascending order
strings.sort(key=len)

# Concatenate all strings
result = ''.join(strings)

print(result)