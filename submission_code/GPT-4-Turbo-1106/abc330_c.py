import math

# Read the input from stdin
D = int(input())

# Initialize the minimum difference to a large number
min_diff = float('inf')

# We only need to check up to the square root of D, since x^2 or y^2 cannot be greater than D
# to minimize the absolute difference.
limit = int(math.sqrt(D)) + 1

for x in range(limit):
    # Calculate the corresponding y^2 based on the current x and D
    y_squared = D - x**2
    # If y_squared is negative, we can't find a non-negative y, so we break the loop
    if y_squared < 0:
        break
    # Calculate y from y_squared. If y_squared is not a perfect square, y will not be an integer,
    # and thus not a valid solution. We only consider y_squared that are perfect squares.
    y = int(math.sqrt(y_squared))
    if y**2 == y_squared:
        # If y_squared is a perfect square, we found a valid pair (x, y)
        # and the difference is 0, which is the minimum possible.
        min_diff = 0
        break
    else:
        # If y_squared is not a perfect square, we calculate the difference
        # for the closest perfect square below and above y_squared.
        lower_y = int(math.sqrt(y_squared))
        upper_y = lower_y + 1
        min_diff = min(min_diff, abs((x**2 + lower_y**2) - D), abs((x**2 + upper_y**2) - D))

# Print the minimum difference
print(min_diff)