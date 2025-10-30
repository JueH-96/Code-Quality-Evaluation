# YOUR CODE HERE
from collections import Counter

# Read input
A = list(map(int, input().split()))

# Count occurrences of each color
color_count = Counter(A)

# Calculate the maximum number of operations
max_operations = sum(count // 2 for count in color_count.values())

# Print the result
print(max_operations)