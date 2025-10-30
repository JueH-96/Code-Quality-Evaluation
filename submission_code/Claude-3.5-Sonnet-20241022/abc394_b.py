# Read N
N = int(input())

# Read N strings and store them in a list
strings = []
for _ in range(N):
    s = input().strip()
    strings.append(s)

# Sort strings by length
sorted_strings = sorted(strings, key=len)

# Concatenate strings in order of increasing length
result = ''.join(sorted_strings)

# Print result
print(result)