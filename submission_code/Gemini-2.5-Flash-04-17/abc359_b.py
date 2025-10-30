# YOUR CODE HERE
# Read N
n = int(input())

# Read the list A
# The list contains 2N integers representing the colors of clothes
# A_i is the color of the person at the i-th position from the left (1-based index).
# We will use 0-based indexing for the list A.
a = list(map(int, input().split()))

# Store the indices (0-based) where each color appears in the list A.
# Since each color from 1 to N appears exactly twice, the dictionary
# color_indices[color] will be a list containing exactly two indices.
color_indices = {}
for i in range(2 * n):
    color = a[i]
    # If this is the first time we see this color, create an empty list for it.
    if color not in color_indices:
        color_indices[color] = []
    # Append the current index to the list for this color.
    color_indices[color].append(i)

# Initialize a counter for colors that satisfy the condition.
count = 0

# Iterate through each color from 1 to N.
# These are the colors we need to check according to the problem statement.
for color in range(1, n + 1):
    # Get the two indices where the current color appears.
    # The problem guarantees that color_indices[color] will have exactly two elements.
    indices = color_indices[color]
    idx1 = indices[0] # The first index of the color
    idx2 = indices[1] # The second index of the color

    # The condition is that there is exactly one person between the two people
    # wearing clothes of color 'color'.
    # If the two people are at 1-based positions p1 and p2, where p1 < p2,
    # the people between them are at positions p1+1, p1+2, ..., p2-1.
    # The number of people between them is (p2 - 1) - (p1 + 1) + 1 = p2 - 1 - p1 - 1 + 1 = p2 - p1 - 1.
    # We want this number to be exactly 1.
    # So, p2 - p1 - 1 = 1, which means p2 - p1 = 2.
    # This is equivalent to |p1 - p2| = 2.
    # Using 0-based indices i1 and i2 (where p = i + 1), we have
    # |(i1 + 1) - (i2 + 1)| = |i1 - i2|.
    # So, the condition is equivalent to checking if the absolute difference
    # between the two 0-based indices is exactly 2.
    if abs(idx1 - idx2) == 2:
        count += 1

# Print the final count of colors that satisfy the condition.
print(count)