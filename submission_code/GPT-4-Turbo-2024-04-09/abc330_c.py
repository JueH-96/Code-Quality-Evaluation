import sys
import math

input = sys.stdin.read
D = int(input().strip())

# We need to find the minimum value of |x^2 + y^2 - D|
# where x and y are non-negative integers.

# We can estimate an upper bound for x and y by considering
# when x^2 + y^2 is around D. Thus, x and y should be around sqrt(D).
max_xy = int(math.sqrt(D)) + 2  # +2 to ensure we cover edge cases

min_diff = float('inf')

# We iterate over possible values of x and y
for x in range(max_xy + 1):
    for y in range(max_xy + 1):
        current_sum = x**2 + y**2
        diff = abs(current_sum - D)
        if diff < min_diff:
            min_diff = diff
        # Since x^2 + y^2 is non-decreasing as x or y increases,
        # if current_sum > D + min_diff, further y will only increase the difference
        if current_sum > D + min_diff:
            break

print(min_diff)