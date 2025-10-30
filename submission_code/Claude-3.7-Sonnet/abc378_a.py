# YOUR CODE HERE
A = list(map(int, input().strip().split()))

# Count the occurrences of each color
color_counts = {}
for color in A:
    color_counts[color] = color_counts.get(color, 0) + 1

# Calculate the number of pairs for each color and sum them
operations = 0
for count in color_counts.values():
    operations += count // 2

print(operations)