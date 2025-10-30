class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        zeros = []
        for idx, c in enumerate(s):
            if c == '0':
                zeros.append(idx)
        m = len(zeros)
        total = 0

        # Handle case when there are no zeros (all 1's)
        if m == 0:
            return n * (n + 1) // 2

        # Handle z=0 case (substrings with 0 zeros)
        # Compute runs of 1's between zeros and at the ends
        total_z0 = 0

        # Before first zero
        first_zero = zeros[0]
        run_length = first_zero
        total_z0 += run_length * (run_length + 1) // 2

        # Between zeros
        for i in range(1, m):
            prev_zero = zeros[i-1]
            curr_zero = zeros[i]
            run_length = curr_zero - prev_zero - 1
            if run_length > 0:
                total_z0 += run_length * (run_length + 1) // 2

        # After last zero
        last_zero = zeros[-1]
        run_length = (n - 1) - last_zero
        total_z0 += run_length * (run_length + 1) // 2

        total += total_z0

        # Now handle z from 1 to sqrt(n)
        max_z = int(n**0.5)
        for z in range(1, max_z + 1):
            L_req = z * z + z
            if L_req > n:
                continue

            current_total = 0
            # Iterate over all groups of exactly z zeros
            for i in range(0, m - z + 1):
                first = zeros[i]
                last = zeros[i + z - 1]
                if i == 0:
                    prev_zero = -1
                else:
                    prev_zero = zeros[i - 1]
                if i + z < m:
                    next_zero = zeros[i + z]
                else:
                    next_zero = n  # beyond the last zero

                left = first - prev_zero
                right = next_zero - last
                minimal_length = last - first + 1

                if minimal_length >= L_req:
                    contribution = left * right
                    current_total += contribution
                else:
                    # Compute using the formula
                    s_start = prev_zero + 1
                    s_end = first
                    e_start = last
                    e_end = next_zero - 1

                    B = s_end
                    A = s_start
                    C = e_start
                    D = e_end
                    E = L_req

                    # Compute first part
                    upper1 = min(D, B + E - 1)
                    if C > upper1:
                        part1 = 0
                    else:
                        a = C
                        b = upper1
                        first_term = a - (E + A - 2)
                        last_term = b - (E + A - 2)
                        num_terms = b - a + 1
                        sum1 = num_terms * (first_term + last_term) // 2
                        part1 = sum1

                    # Compute second part
                    lower2 = max(C, B + E)
                    if lower2 > D:
                        part2 = 0
                    else:
                        num_terms2 = D - lower2 + 1
                        term = (B - A + 1)
                        part2 = term * num_terms2

                    total_part = part1 + part2
                    current_total += total_part

            total += current_total

        return total