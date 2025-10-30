import sys

# Read the number of bean types, N.
# sys.stdin.readline() is used for faster I/O in competitive programming,
# which can be important for large inputs.
try:
    n_str = sys.stdin.readline()
    # Handle empty input case
    if not n_str:
        n = 0
    else:
        n = int(n_str)
except (ValueError, IndexError):
    n = 0

# This dictionary will store the minimum deliciousness found for each color.
# Key: color ID
# Value: minimum deliciousness for that color
min_deliciousness_for_color = {}

# Process each of the N bean types.
for _ in range(n):
    # Read the deliciousness (A_i) and color (C_i) for the current bean type.
    try:
        deliciousness, color = map(int, sys.stdin.readline().split())
    except (ValueError, IndexError):
        # Skip malformed lines
        continue

    # Check if we have encountered this color before.
    if color in min_deliciousness_for_color:
        # If we have, we update the stored minimum deliciousness for this color
        # only if the current bean's deliciousness is smaller.
        # This ensures we always store the minimum deliciousness for each color.
        min_deliciousness_for_color[color] = min(min_deliciousness_for_color[color], deliciousness)
    else:
        # If this is the first time we're seeing this color, we record its
        # deliciousness as the initial minimum for that color.
        min_deliciousness_for_color[color] = deliciousness

# The problem asks to maximize the minimum possible deliciousness.
# For each color, the minimum possible deliciousness is the value we've stored
# in our dictionary.
# To maximize this, we simply find the maximum value among all the minimums
# we've collected.
# The problem constraints (N >= 1) guarantee that the dictionary will not be empty,
# so calling max() on its values is safe.
if min_deliciousness_for_color:
    result = max(min_deliciousness_for_color.values())
    # Print the final answer to standard output.
    print(result)