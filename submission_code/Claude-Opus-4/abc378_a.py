# YOUR CODE HERE
# Read input
balls = list(map(int, input().split()))

# Count frequency of each color
color_count = {}
for ball in balls:
    color_count[ball] = color_count.get(ball, 0) + 1

# Calculate maximum operations
max_operations = 0
for count in color_count.values():
    max_operations += count // 2

print(max_operations)