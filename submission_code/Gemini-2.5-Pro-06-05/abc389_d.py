import sys
import math

def solve():
    """
    Reads radius R and calculates the number of squares completely contained within the circle.
    """
    try:
        R_str = sys.stdin.readline()
        if not R_str:
            return
        R = int(R_str)
    except (IOError, ValueError):
        # Handle empty input or invalid format
        return

    # R^2 can be large, up to 10^12. Python handles large integers automatically.
    R_sq = R * R

    # The total count is found by summing the number of valid squares in each column `i`.
    # By symmetry, the count for column `i` is the same as for `-i`.
    # Total count = (count for i=0) + 2 * (sum of counts for i > 0)
    
    # Let max_j(i) be the largest non-negative integer `j` that satisfies the condition
    # for a given `i`: (|i| + 0.5)^2 + (j + 0.5)^2 <= R^2.
    # This implies j <= sqrt(R^2 - (|i| + 0.5)^2) - 0.5.
    # So, max_j(i) = floor(sqrt(R^2 - (|i| + 0.5)^2) - 0.5).
    
    # For a given `i`, the valid `j` values are -max_j(i), ..., 0, ..., max_j(i).
    # The number of such values is 2 * max_j(i) + 1.

    # 1. Sum the counts for all positive i columns (i from 1 to R-1).
    count_for_positive_i_columns = 0
    for i in range(1, R):
        # We need R^2 - (i+0.5)^2 >= (0+0.5)^2 = 0.25 for any solution j to exist.
        y_sq_val = R_sq - (i + 0.5)**2
        if y_sq_val < 0.25:
            # If the condition is not met for this i, it won't be for any larger i.
            break
        
        max_j = math.floor(math.sqrt(y_sq_val) - 0.5)
        # Number of valid squares in column `i` is 2 * max_j + 1.
        count_for_positive_i_columns += (2 * max_j + 1)

    # 2. Total count from columns i > 0 and i < 0 is twice the sum for i > 0.
    total_count = 2 * count_for_positive_i_columns

    # 3. Add the contribution from the column i = 0.
    y_sq_val_0 = R_sq - 0.25
    # For R>=1, y_sq_val_0 is always >= 0.75, so its sqrt is real.
    max_j0 = math.floor(math.sqrt(y_sq_val_0) - 0.5)
    count_for_i_0 = 2 * max_j0 + 1
    
    total_count += count_for_i_0
    
    print(total_count)

solve()