# YOUR CODE HERE
n = int(input())
strings = []
for _ in range(n):
    strings.append(input().strip())

# Sort strings by length
strings.sort(key=len)

# Concatenate all strings
result = ''.join(strings)

print(result)