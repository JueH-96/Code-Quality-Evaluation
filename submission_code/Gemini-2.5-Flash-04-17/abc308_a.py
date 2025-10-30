# YOUR CODE HERE
import sys

# Read the line from standard input
line = sys.stdin.readline()

# Split the line by spaces and convert each part to an integer
# Assumes exactly 8 integers as per problem description and constraints.
# If the input format could vary, additional checks (e.g., len(s_list) == 8)
# would be necessary, but for this problem, it's typically guaranteed.
s_list = list(map(int, line.split()))

# Condition 1: The sequence is monotonically non-decreasing.
# This means S_1 <= S_2 <= ... <= S_8.
# We check s_list[i] <= s_list[i+1] for i from 0 to 6.
# The all() function with a generator expression is a concise way to check if
# this condition holds for all adjacent pairs.
is_non_decreasing = all(s_list[i] <= s_list[i+1] for i in range(7))

# Conditions 2 & 3: Each element must be between 100 and 675 inclusive,
# AND each element must be a multiple of 25.
# We check 100 <= s <= 675 and s % 25 == 0 for each element s in the list.
# Again, all() is used to check if this combined condition holds for all elements.
is_valid_elements = all(100 <= s <= 675 and s % 25 == 0 for s in s_list)

# The sequence satisfies all conditions if and only if it is both
# non-decreasing AND all its elements meet the range and multiple requirements.
if is_non_decreasing and is_valid_elements:
    print("Yes")
else:
    print("No")