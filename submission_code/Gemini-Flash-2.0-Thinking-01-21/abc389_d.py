import sys
import math

def solve():
    R = int(sys.stdin.readline())
    R_float = float(R)
    R_sq = R_float**2

    total_count = 0

    # Iterate through possible integer values of i from 0 upwards.
    # Due to symmetry, we only need to consider i >= 0.
    # The range of j will be calculated based on |i|.
    i = 0
    while True:
        # Calculate the term (i + 0.5)^2
        # Using float(i) + 0.5 for precision
        i_plus_half = float(i) + 0.5
        i_term_sq = i_plus_half**2

        # Calculate the remaining squared radius needed for the j term.
        # R^2 - (i + 0.5)^2
        rem_R2 = R_sq - i_term_sq

        # If rem_R2 is negative (within a small tolerance for floating point error),
        # it means (i + 0.5)^2 > R^2, so for this i (and any larger i),
        # (i + 0.5)^2 + (|j| + 0.5)^2 will always be > R^2 for any integer j.
        # Thus, no squares with this |i| or larger are completely contained. We can stop.
        # Using a small negative tolerance like -1e-9 to account for potential precision issues
        # near the boundary where rem_R2 is mathematically 0 but numerically slightly negative.
        if rem_R2 < -1e-9:
            break

        # Ensure the value inside sqrt is non-negative, even if rem_R2 is slightly negative due to FP error.
        rem_R2 = max(0.0, rem_R2)

        # We need (|j| + 0.5)^2 <= rem_R2
        # |j| + 0.5 <= sqrt(rem_R2)
        # |j| <= sqrt(rem_R2) - 0.5
        threshold_j = math.sqrt(rem_R2) - 0.5

        # We need integer j such that |j| <= threshold_j.
        # Let K be the maximum integer such that K <= threshold_j. K = floor(threshold_j).
        # If K < 0, there are no integers j satisfying |j| <= threshold_j.
        # If K >= 0, the integers j satisfying |j| <= threshold_j are 0, +/-1, +/-2, ..., +/-K.
        # The count of such integers is 1 (for j=0) + 2*K (for +/-1 to +/-K) = 2*K + 1.
        # If K < 0, 2*K+1 will be negative. We need the count to be 0 in this case.
        # So, the number of integer j values for this i is max(0, 2 * K + 1).
        K = math.floor(threshold_j)
        num_j_for_i = max(0, 2 * K + 1)

        # Add the count for this i to the total.
        # If i = 0, this accounts for the column of squares along the y-axis (including the origin).
        # If i > 0, this accounts for the column of squares with x-coordinate i.
        # By symmetry, the column of squares with x-coordinate -i has the same number of contained squares.
        # So for i > 0, we add 2 * num_j_for_i to the total count.
        if i == 0:
            total_count += num_j_for_i
        else:
            total_count += 2 * num_j_for_i

        i += 1

    print(total_count)

solve()