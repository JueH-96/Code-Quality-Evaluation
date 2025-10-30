import math
import sys

def solve():
    D = int(sys.stdin.readline())

    # Initialize min_diff with the absolute difference for x=0, y=0.
    # This is |0^2 + 0^2 - D| = D.
    min_diff = D 

    # Determine an upper bound for x.
    # If x^2 + y^2 is to be close to D, then x cannot be arbitrarily large.
    # If x is such that x^2 > 2*D, then x^2 + y^2 - D > D, which would be
    # larger than our initial min_diff (which is D).
    # So, we only need to check x up to approximately sqrt(2*D).
    # math.isqrt(N) computes floor(sqrt(N)).
    # We add a small constant (e.g., 5) for robustness, although isqrt is precise.
    max_x_to_check = math.isqrt(2 * D) + 5

    # Iterate through possible non-negative integer values for x.
    for x in range(max_x_to_check + 1):
        x_sq = x * x

        # Optimization: If x_sq alone (with y=0) is already greater than D,
        # and the difference (x_sq - D) is greater than or equal to the
        # current minimum difference found, then any larger x (and any y >= 0)
        # will only yield a larger or equal absolute difference.
        # So we can stop searching.
        # This condition is crucial for performance with large D.
        # Note: `x_sq - D >= min_diff` implicitly handles `x_sq > D`
        # because `min_diff` is always non-negative.
        if x_sq - D >= min_diff:
            break
        
        # Calculate the target value for y_sq.
        # We want y_sq to be close to D - x_sq.
        target_y_sq = D - x_sq

        if target_y_sq < 0:
            # If D - x_sq is negative, it means x_sq is already greater than D.
            # To minimize |x^2 + y^2 - D|, we must choose y = 0.
            # In this case, x^2 + y^2 - D becomes x_sq - D.
            # Since x_sq is already > D, this difference will be positive.
            min_diff = min(min_diff, x_sq - D)
        else:
            # target_y_sq is non-negative. Find y values near sqrt(target_y_sq).
            y_floor = math.isqrt(target_y_sq)

            # Check the sum for y = y_floor
            y_floor_sq = y_floor * y_floor
            current_sum_1 = x_sq + y_floor_sq
            min_diff = min(min_diff, abs(current_sum_1 - D))

            # Check the sum for y = y_floor + 1.
            # This covers cases where (y_floor+1)^2 might be closer to target_y_sq
            # (i.e., (y_floor+1)^2 might be slightly greater than target_y_sq,
            # but y_floor^2 is slightly less than target_y_sq).
            y_ceil = y_floor + 1
            y_ceil_sq = y_ceil * y_ceil
            current_sum_2 = x_sq + y_ceil_sq
            min_diff = min(min_diff, abs(current_sum_2 - D))
            
            # If min_diff becomes 0, it's the absolute minimum, so we can stop.
            if min_diff == 0:
                break

    sys.stdout.write(str(min_diff) + "
")

# Call the solve function to run the program.
solve()