from collections import Counter

# Read the input as a list of integers
colors = list(map(int, input().split()))

# Count the frequency of each color
color_counts = Counter(colors)

# Calculate the maximum number of operations
max_operations = sum(count // 2 for count in color_counts.values())

# Output the result
print(max_operations)