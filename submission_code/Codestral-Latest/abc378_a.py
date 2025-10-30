# YOUR CODE HERE
from collections import Counter

# Read the input
A = list(map(int, input().split()))

# Count the frequency of each color
color_count = Counter(A)

# The maximum number of operations is the maximum frequency of any color
max_operations = max(color_count.values())

# Print the result
print(max_operations)