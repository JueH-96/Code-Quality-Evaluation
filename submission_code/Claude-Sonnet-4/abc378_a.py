# YOUR CODE HERE
from collections import Counter

# Read input
colors = list(map(int, input().split()))

# Count frequency of each color
color_count = Counter(colors)

# Calculate maximum operations
total_operations = 0
for count in color_count.values():
    total_operations += count // 2

print(total_operations)