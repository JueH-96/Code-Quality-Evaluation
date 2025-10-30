# YOUR CODE HERE
from collections import Counter

# Read input from stdin
balls = list(map(int, input().split()))

# Count the occurrences of each color
color_counts = Counter(balls)

# Calculate the maximum number of operations
max_operations = sum(count // 2 for count in color_counts.values())

# Print the result
print(max_operations)