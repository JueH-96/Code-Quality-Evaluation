n = int(input())
strings = [input().strip() for _ in range(n)]
# Sort the strings based on their lengths
sorted_strings = sorted(strings, key=lambda x: len(x))
# Concatenate them
result = ''.join(sorted_strings)
print(result)