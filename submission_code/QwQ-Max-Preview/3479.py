class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        zeros_pos = []
        runs_of_ones = []
        current_run = None
        
        for i, char in enumerate(s):
            if char == '0':
                zeros_pos.append(i)
                if current_run is not None:
                    runs_of_ones.append(current_run)
                    current_run = None
            else:
                if current_run is None:
                    current_run = {'start': i, 'end': i}
                else:
                    current_run['end'] = i
        if current_run is not None:
            runs_of_ones.append(current_run)
        
        count_z0 = 0
        for run in runs_of_ones:
            k = run['end'] - run['start'] + 1
            count_z0 += k * (k + 1) // 2
        
        m = len(zeros_pos)
        total = count_z0
        
        if m == 0:
            return total
        
        max_z = int((((4 * n + 1) ** 0.5) - 1) // 2)
        max_z = min(max_z, m)
        
        for z in range(1, max_z + 1):
            if z > m:
                continue
            for i in range(m - z + 1):
                l = zeros_pos[i]
                r = zeros_pos[i + z - 1]
                
                prev_zero = zeros_pos[i - 1] if i > 0 else -1
                next_zero = zeros_pos[i + z] if (i + z) < m else n
                
                a = prev_zero + 1
                b = next_zero
                
                required_len = z * (z + 1)
                
                s_max = r - required_len + 1
                start_part1_start = a
                start_part1_end = min(l, s_max)
                
                if start_part1_start > start_part1_end:
                    count_part1 = 0
                else:
                    number_part1 = start_part1_end - start_part1_start + 1
                    count_part1 = number_part1 * (b - r)
                
                start_part2_start = max(a, s_max + 1)
                start_part2_end = l
                if start_part2_start > start_part2_end:
                    count_part2 = 0
                else:
                    start_max_valid = min(start_part2_end, b + 1 - required_len)
                    if start_part2_start > start_max_valid:
                        count_part2 = 0
                    else:
                        number_part2 = start_max_valid - start_part2_start + 1
                        first_term = (b + 1 - required_len) - start_part2_start
                        last_term = (b + 1 - required_len) - start_max_valid
                        sum_part2 = number_part2 * (first_term + last_term) // 2
                        count_part2 = max(sum_part2, 0)
                
                total += count_part1 + count_part2
        
        return total