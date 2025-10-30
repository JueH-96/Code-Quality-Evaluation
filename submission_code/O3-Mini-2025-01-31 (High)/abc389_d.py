def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    R = int(data[0])
    # A square whose center is at (i,j) has four corners:
    #   (i ± 0.5, j ± 0.5).
    # It is completely inside a circle of radius R (centered at 0,0) if every
    # corner satisfies:
    #   (|i| + 0.5)² + (|j| + 0.5)² ≤ R².
    #
    # Multiplying by 4, the condition is equivalent to:
    #   (2|i| + 1)² + (2|j| + 1)² ≤ 4*R².
    #
    # Because the condition depends only on |i| and |j|, we count solutions
    # in the first quadrant (i ≥ 0, j ≥ 0) and then use symmetry.
    #
    # For a given nonnegative integer i, the condition becomes:
    #   (2*i + 1)² + (2*j + 1)² ≤ 4*R².
    # Solving for j, we need:
    #   (2*j + 1)² ≤ 4*R² - (2*i + 1)².
    #
    # Then the maximum j (denoted j_max) that satisfies this is:
    #   j_max = floor((sqrt(4*R² - (2*i + 1)²) - 1) / 2).
    # The number of j-values is (j_max + 1), provided that the right-hand side is at least 0.
    #
    # In the full plane the reflection symmetry gives these weights:
    #   • (0, 0) appears only once.
    #   • (0, j) with j > 0 appears twice (j and -j).
    #   • (i, 0) with i > 0 appears twice.
    #   • (i, j) with i > 0 and j > 0 appears in four quadrants.
    #
    # So if for a given i we set:
    #   count = (j_max + 1)    (with j in [0, j_max]),
    # then the contribution from that row is:
    #   if i == 0:       contribution = 1 (for j=0) + 2*(count - 1)
    #   if i >  0:       contribution = 2 (for j=0) + 4*(count - 1)
    #
    # Finally, note that i can range only while (2*i+1) <= 2R, i.e.
    #   i ≤ R - 0.5.
    # Hence we loop i from 0 to floor(R-0.5).

    R2 = 4 * R * R
    i_max = int(R - 0.5)
    total = 0

    for i in range(i_max + 1):
        # (2*i+1)² contributes from the i-th step:
        lhs = (2 * i + 1) ** 2
        # The remaining term for j must satisfy (2*j+1)² ≤ R2 - lhs.
        rem = R2 - lhs
        if rem < 1:
            break  # No valid j values for this (and larger) i.
        # We want the largest integer j for which (2*j+1)² ≤ rem.
        # Rearranging: 2*j + 1 ≤ sqrt(rem)   →   j ≤ (sqrt(rem) - 1) / 2.
        # Using math.isqrt gives the integer floor of sqrt(rem).
        fs = math.isqrt(rem)
        j_max = (fs - 1) // 2  # j_max = floor((sqrt(rem)-1)/2)
        # The number of valid j (with j ≥ 0) is (j_max + 1).
        count_for_this_i = j_max + 1

        # Now account for symmetry:
        if i == 0:
            # j = 0 counts only once.
            # For j ≥ 1, the positive and negative j values give 2 squares each.
            row_contrib = 1 + 2 * (count_for_this_i - 1)
        else:
            # For i > 0, (i, 0) appears twice and any (i, j) with j ≥ 1 appears in 4 quadrants.
            row_contrib = 2 + 4 * (count_for_this_i - 1)
        total += row_contrib

    sys.stdout.write(str(total))

if __name__ == '__main__':
    main()