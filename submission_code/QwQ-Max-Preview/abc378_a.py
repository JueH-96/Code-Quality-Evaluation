# Read the input as a list of integers
balls = list(map(int, input().split()))

# Count the occurrences of each color
from collections import Counter
color_counts = Counter(balls)

# Calculate the total number of operations
total_operations = 0
for count in color_counts.values():
    total_operations += count // 2

# Output the result
print(total_operations)