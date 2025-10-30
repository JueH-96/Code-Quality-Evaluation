def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    W = int(input_data[1])
    wv = input_data[2:]

    w = [0]*N
    v = [0]*N
    idx = 0
    for i in range(N):
        w[i] = int(wv[idx])
        v[i] = int(wv[idx+1])
        idx += 2

    NEG_INF = -10**18  # A large negative number for "impossible" states

    # dp[c] will hold the best "total happiness" achievable using
    # the first i types (in the loop) with capacity exactly c.
    # Start with dp[0] = 0, and all others -∞ (not achievable).
    dp = [NEG_INF]*(W+1)
    dp[0] = 0

    def intersect(line1, line2):
        # line: [m, b, x] represents y = m*x + b, valid for x >= that line's x-bound
        # We only need the intersection x-value of line1 and line2.
        # For maximum hull with different slopes, intersection x = (b2 - b1)/(m1 - m2).
        return (line2[1] - line1[1]) / (line1[0] - line2[0])

    def solve_for_one_type(old_dp, wt, val):
        """
        Given old_dp[] (dp from previous types),
        compute new_dp[] after possibly taking items of type that has weight=wt, value=val.

        The 'happiness' for taking k items of this type = k*val - k^2,
        but we combine with the old dp by capacity.  We handle each residue mod wt separately
        using a "convex-hull-trick"-like optimization to keep it at O(W) per type.
        """
        new_dp = [NEG_INF]*(W+1)

        for r in range(wt):
            # We'll process all capacities c with c % wt == r in one sweep.
            # old_r[x] = old_dp[r + x*wt], for x = 0..maxJ where maxJ = floor((W-r)/wt).
            maxJ = (W - r) // wt
            old_r = [old_dp[r + j*wt] for j in range(maxJ + 1)]
            out_r = [NEG_INF]*(maxJ + 1)

            hull = []  # Will store lines as [m, b, x_intersect]
            head = 0   # pointer for answering queries in ascending order of x

            # Add the line corresponding to x=0, if that state is valid
            # slope = 0, intercept = old_r[0] - (0^2) = old_r[0].
            if old_r[0] != NEG_INF:
                hull.append([0, old_r[0], float('-inf')])

            # Now for j in [0..maxJ], we want:
            # new_r[j] = max_{x in [0..j]} [old_r[x] + (j-x)*val - (j-x)^2 ]
            # which can be rearranged into a "convex hull" form:
            # new_r[j] = j*val - j^2 + max_x [old_r[x] - x^2 + x*(2j - val)].
            # So each "x" adds a "line" with slope = x, intercept = old_r[x] - x^2.
            # We do queries for xQ = 2j - val in ascending order of j.

            for j in range(maxJ + 1):
                # The query's x coordinate in that linear form is (2*j - val).
                xq = 2*j - val

                # Move pointer if needed: we pick whichever line of the hull is best for xq
                # The hull is built so that for ascending xq, we can "pop" forward.
                while head + 1 < len(hull) and hull[head+1][2] <= xq:
                    head += 1

                if len(hull) > 0:
                    m, b, _ = hull[head]
                    best_val = m*xq + b
                    out_r[j] = j*val - j*j + best_val

                # After using line slopes <= j, we then add slope=j+1 for the next iteration
                # as long as j+1 <= maxJ.
                if j < maxJ:
                    # old_r[j+1] might be -∞ (meaning not reachable); skip if so.
                    if old_r[j+1] != NEG_INF:
                        m_next = j + 1
                        b_next = old_r[j+1] - (j+1)*(j+1)
                        # Insert line y = m_next*x + b_next in ascending slope order
                        # We must pop lines from the hull if the new line
                        # intersects them "earlier" than they intersect each other.
                        # Also handle the case of duplicate slopes.
                        if hull and hull[-1][0] == m_next:
                            # same slope; keep only the one with better intercept (bigger for max)
                            if b_next <= hull[-1][1]:
                                # Not better than the last line
                                continue
                            else:
                                hull.pop()  # remove the old line with the same slope

                        # Now we do the usual intersection-based removal
                        new_line = [m_next, b_next, 0.0]
                        while len(hull) >= 2:
                            inter1 = intersect(hull[-2], hull[-1])
                            inter2 = intersect(hull[-2], new_line)
                            if inter2 <= inter1:
                                hull.pop()
                            else:
                                break

                        if not hull:
                            new_line[2] = float('-inf')
                        else:
                            new_line[2] = intersect(hull[-1], new_line)
                        hull.append(new_line)

            # Write the results for this residue back to new_dp
            for j in range(maxJ + 1):
                new_dp[r + j*wt] = out_r[j]

        return new_dp

    # Process each type in turn
    for i in range(N):
        dp = solve_for_one_type(dp, w[i], v[i])

    # The answer is the maximum dp[c] for c in [0..W].
    print(max(dp))


# Do not forget to call main!