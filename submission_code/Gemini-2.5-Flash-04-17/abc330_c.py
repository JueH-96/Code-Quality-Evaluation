# YOUR CODE HERE
import math

# Read input
D = int(input())

# Initialize minimum difference
# A safe initial value is the difference for x=0, y=0
min_diff = D

# Iterate through possible values of x
# The loop terminates when x^2 is sufficiently large
x = 0
while True:
    x_sq = x * x

    # Consider the point (x, 0). The sum of squares is x_sq + 0^2 = x_sq.
    # Calculate the difference |x_sq - D|.
    # This handles the case when target_ysq < 0 (i.e., x_sq > D), where the minimum y is 0.
    current_diff_y0 = abs(x_sq - D)
    min_diff = min(min_diff, current_diff_y0)

    # If x_sq exceeds D + min_diff, then for this x and any y >= 0,
    # x^2 + y^2 >= x^2 > D + min_diff.
    # So, x^2 + y^2 - D > min_diff. The difference is strictly greater than min_diff.
    # We can stop iterating x, as increasing x further will only increase x^2 + y^2.
    # This check must happen *after* calculating min_diff for (x, 0).
    # Use `x_sq > D + min_diff` as the break condition.
    # If x_sq == D + min_diff, |x_sq - D| = min_diff. This case is fine, and the loop continues.
    # If x_sq > D + min_diff, we break.
    if x_sq > D + min_diff:
        break

    # Consider cases where y > 0. We need y^2 to be close to D - x_sq.
    # This is only relevant if D - x_sq is non-negative.
    target_ysq = D - x_sq

    if target_ysq >= 0:
        # Find y such that y^2 is close to target_ysq.
        # The candidates for y are floor(sqrt(target_ysq)) and floor(sqrt(target_ysq)) + 1.
        y1 = int(math.sqrt(target_ysq)) # floor(sqrt)

        # Candidate 1: y = y1
        # y1 is guaranteed to be non-negative since target_ysq >= 0.
        val1 = x_sq + y1 * y1
        diff1 = abs(val1 - D)
        min_diff = min(min_diff, diff1)

        # Candidate 2: y = y1 + 1
        y2 = y1 + 1
        # y2 is guaranteed to be non-negative since y1 >= 0.
        val2 = x_sq + y2 * y2
        diff2 = abs(val2 - D)
        min_diff = min(min_diff, diff2)

    # If target_ysq < 0 (i.e., x_sq > D), the only relevant non-negative y is y=0.
    # This case is handled by `current_diff_y0 = abs(x_sq - D)`.
    # We don't need an else block here.

    # Move to the next x
    x += 1

# Print the result
print(min_diff)