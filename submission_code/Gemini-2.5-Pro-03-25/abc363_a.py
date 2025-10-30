# YOUR CODE HERE
import sys

# Read the input rating R from standard input
R = int(sys.stdin.readline())

# Determine the current category based on the rating R.
# The thresholds for increasing the number of '^' are 100, 200, 300.
# We need to find the smallest multiple of 100 that is strictly greater than R.

# If 1 <= R <= 99, the next threshold is 100.
# If 100 <= R <= 199, the next threshold is 200.
# If 200 <= R <= 299, the next threshold is 300.

# We can calculate the next threshold using integer division.
# Dividing R by 100 gives the current "hundreds" block (0 for 1-99, 1 for 100-199, 2 for 200-299).
# Adding 1 gives the index of the next hundreds block.
# Multiplying by 100 gives the rating at the start of the next block (the target threshold).
target_rating = (R // 100 + 1) * 100

# The minimum increase required is the difference between the target rating and the current rating R.
required_increase = target_rating - R

# Print the result to standard output
print(required_increase)