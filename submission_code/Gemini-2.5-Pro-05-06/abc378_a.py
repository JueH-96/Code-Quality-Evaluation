import collections

# Read the four space-separated integers representing ball colors
# and convert them into a list of integers.
balls = list(map(int, input().split()))

# Use collections.Counter to count the frequency of each ball color.
# For example, if balls = [2, 1, 2, 1], color_counts will be Counter({2: 2, 1: 2}).
color_counts = collections.Counter(balls)

# Initialize the total number of operations.
num_operations = 0

# For each color, the number of pairs we can form is its count divided by 2 (integer division).
# Sum these up to get the total number of operations.
# For example, if color_counts.values() are [2, 2] (from Counter({2:2, 1:2})),
# then num_operations will be (2//2) + (2//2) = 1 + 1 = 2.
# If color_counts.values() are [3, 1] (from Counter({4:3, 1:1})),
# then num_operations will be (3//2) + (1//2) = 1 + 0 = 1.
for count in color_counts.values():
    num_operations += count // 2

# Print the result.
print(num_operations)

# A more compact way to write the calculation part:
# num_operations = sum(count // 2 for count in color_counts.values())
# print(num_operations)