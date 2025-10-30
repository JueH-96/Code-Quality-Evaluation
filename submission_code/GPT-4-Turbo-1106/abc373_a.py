# Read the strings from standard input
strings = [input().strip() for _ in range(12)]

# Count the number of strings whose length matches their index (1-indexed)
count = sum(len(strings[i]) == i + 1 for i in range(12))

# Print the result
print(count)