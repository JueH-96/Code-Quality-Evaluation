from collections import Counter

# Read the input as a list of integers
a = list(map(int, input().split()))

# Count the occurrences of each color
counts = Counter(a)

# Calculate the total number of operations
total = sum(v // 2 for v in counts.values())

# Print the result
print(total)