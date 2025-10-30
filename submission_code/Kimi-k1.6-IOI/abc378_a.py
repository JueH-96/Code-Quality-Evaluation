import collections

# Read the input as a list of integers
balls = list(map(int, input().split()))

# Count the occurrences of each color
color_counts = collections.Counter(balls)

# Calculate the maximum number of operations by summing the integer division of each count by 2
max_operations = sum(count // 2 for count in color_counts.values())

# Output the result
print(max_operations)