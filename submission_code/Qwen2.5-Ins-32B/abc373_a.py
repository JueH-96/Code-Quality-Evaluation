# Read the 12 strings from stdin
strings = [input().strip() for _ in range(12)]

# Count how many strings have a length equal to their index + 1
count = sum(1 for i, s in enumerate(strings, start=1) if len(s) == i)

# Print the result to stdout
print(count)