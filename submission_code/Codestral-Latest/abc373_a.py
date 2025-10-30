# YOUR CODE HERE
# Read the 12 strings from standard input
strings = []
for _ in range(12):
    strings.append(input().strip())

# Count the number of strings where the length matches the index (1-based)
count = sum(1 for i, s in enumerate(strings, start=1) if len(s) == i)

# Print the result
print(count)