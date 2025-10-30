import sys
from collections import Counter

# Read input from stdin
input_data = sys.stdin.read().strip()
A = list(map(int, input_data.split()))

# Count the frequency of each color
color_count = Counter(A)

# Calculate the maximum number of operations
max_operations = sum(count // 2 for count in color_count.values())

# Print the result to stdout
print(max_operations)