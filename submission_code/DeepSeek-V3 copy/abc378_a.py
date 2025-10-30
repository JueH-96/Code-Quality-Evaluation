# YOUR CODE HERE
from collections import Counter

# Read the input as a list of integers
balls = list(map(int, input().split()))

# Count the frequency of each color
color_counts = Counter(balls)

# Calculate the number of operations
operations = 0
for count in color_counts.values():
    operations += count // 2

# Print the result
print(operations)