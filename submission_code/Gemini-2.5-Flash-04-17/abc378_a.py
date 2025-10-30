import sys

# Read the input line from standard input.
# The input is expected to be a single line containing four space-separated integers.
# Use sys.stdin.readline() for potentially faster input reading in competitive programming.
line = sys.stdin.readline().split()

# Convert the list of string parts obtained from splitting the line into a list of integers.
# We assume the input format is strictly followed and contains exactly four integers as per constraints.
colors = [int(x) for x in line]

# Count the frequency of each color.
# The problem specifies that colors are integers between 1 and 4, inclusive.
# A frequency array (list in Python) is a suitable data structure for this.
# We create a list of size 5. Index 'i' will store the count of color 'i'.
# Indices 1, 2, 3, and 4 correspond to colors 1, 2, 3, and 4. Index 0 is unused.
counts = [0] * 5

# Iterate through the list of input ball colors to populate the frequency counts.
for color in colors:
    # Based on the problem constraints, 'color' will always be an integer from 1 to 4.
    # We increment the count at the list index corresponding to the color value.
    counts[color] += 1

# Calculate the maximum number of operations possible.
# An operation consists of choosing two balls of the same color and removing them.
# This is equivalent to forming pairs of balls of the same color and discarding each pair.
# For any given color 'i', if there are 'counts[i]' balls of that color,
# we can form 'counts[i] // 2' complete pairs using integer division.
# Each complete pair allows one operation.
# The maximum total number of operations is the sum of the maximum pairs possible for each distinct color,
# as operations on one color do not affect the counts of other colors.
max_operations = 0
# Iterate through the possible color values, which are 1, 2, 3, and 4.
for i in range(1, 5):
    # For each color 'i', calculate the number of pairs (counts[i] // 2) and add this to the total.
    # Integer division (//) ensures we only count complete pairs.
    max_operations += counts[i] // 2

# Print the final calculated maximum number of operations to standard output.
print(max_operations)