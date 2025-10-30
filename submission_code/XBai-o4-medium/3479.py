class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        zeros = []
        n = len(s)
        for i, c in enumerate(s):
            if c == '0':
                zeros.append(i)
        m = len(zeros)
        total = 0
        
        # Handle Z=0 case
        run_length = 0
        for c in s:
            if c == '1':
                run_length += 1
                total += run_length
            else:
                run_length = 0
        
        # Now handle Z >=1
        max_z = int(((1 + 8 * n)**0.5 - 1) // 2)
        for Z in range(1, max_z + 1):
            R = Z * Z + Z
            K = R - 1  # since s_end - s_start >= R-1
            # For each group of Z consecutive zeros
            for i in range(m - Z + 1):
                current_group_start = zeros[i]
                current_group_end = zeros[i + Z - 1]
                # previous zero is zeros[i-1] if i > 0 else -1
                prev_zero = zeros[i-1] if i > 0 else -1
                # next zero is zeros[i + Z] if i + Z < m else n
                if i + Z < m:
                    next_zero = zeros[i + Z]
                else:
                    next_zero = n
                # left_start and left_end
                left_start = prev_zero + 1
                left_end = current_group_start
                # right_start and right_end
                right_start = current_group_end
                right_end = next_zero - 1
                # Now compute count_valid
                A = left_start
                B = left_end
                C = right_start
                D = right_end
                # calculate lower = max(C, A + K)
                lower = max(C, A + K)
                if lower > D:
                    continue
                # end_threshold = B + K
                end_threshold = B + K
                upper_part1 = min(D, end_threshold - 1)
                # part1: from lower to upper_part1
                a_val = lower - (A + K - 1)
                if a_val < 0:
                    a_val = 0
                b_val = upper_part1 - (A + K - 1)
                n_terms_part1 = upper_part1 - lower + 1
                if n_terms_part1 <= 0:
                    sum_part1 = 0
                else:
                    sum_part1 = (a_val + b_val) * n_terms_part1 // 2
                # part2: from end_threshold to D
                sum_part2 = 0
                if end_threshold <= D:
                    count_part2 = D - end_threshold + 1
                    sum_part2 = count_part2 * (B - A + 1)
                current_count = sum_part1 + sum_part2
                total += current_count
        return total