class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        zeros = [i for i, c in enumerate(s) if c == '0']
        m = len(zeros)
        total = 0
        
        # Handle z=0 case: substrings with no zeros
        current = 0
        sum_z0 = 0
        for c in s:
            if c == '1':
                current += 1
            else:
                sum_z0 += current * (current + 1) // 2
                current = 0
        sum_z0 += current * (current + 1) // 2
        total += sum_z0
        
        # Handle z >= 1 cases
        for z in range(1, m + 1):
            required_min_length = z * (z + 1)
            if required_min_length > len(s):
                continue
            num_groups = m - z + 1
            if num_groups <= 0:
                continue
            for i in range(num_groups):
                a = zeros[i]
                b = zeros[i + z - 1]
                # Determine previous and next zero positions
                prev_zero = zeros[i - 1] if i > 0 else -1
                next_zero = zeros[i + z] if (i + z) < m else len(s)
                s_start = prev_zero + 1
                s_end = a
                e_start = b
                e_end = next_zero - 1
                min_length_group = b - a + 1
                if min_length_group >= required_min_length:
                    count = (s_end - s_start + 1) * (e_end - e_start + 1)
                    total += count
                else:
                    # Calculate part1
                    part1_start_threshold = e_start - z * (z + 1) + 1
                    part1_start_min = s_start
                    part1_start_max = min(part1_start_threshold, s_end)
                    if part1_start_min > part1_start_max:
                        part1 = 0
                    else:
                        num_starts_part1 = part1_start_max - part1_start_min + 1
                        if e_end >= e_start:
                            part1_ends = e_end - e_start + 1
                            part1 = num_starts_part1 * part1_ends
                        else:
                            part1 = 0
                    # Calculate part2
                    part2_start_min = part1_start_max + 1
                    part2_start_max = s_end
                    part2_start_max_possible = e_end - z * (z + 1) + 1
                    part2_start_max = min(part2_start_max, part2_start_max_possible)
                    if part2_start_min > part2_start_max:
                        part2 = 0
                    else:
                        a_part2 = part2_start_min
                        b_part2 = part2_start_max
                        C = e_end - z * (z + 1) + 2
                        num_terms = b_part2 - a_part2 + 1
                        sum_C = C * num_terms
                        sum_start = (a_part2 + b_part2) * num_terms // 2
                        part2 = sum_C - sum_start
                    total += part1 + part2
        return total