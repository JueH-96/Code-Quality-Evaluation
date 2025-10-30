class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        pos = []
        for i, c in enumerate(s):
            if c == '0':
                pos.append(i)
        total = 0
        # Handle z=0: count runs of '1's
        current_run = 0
        for c in s:
            if c == '1':
                current_run += 1
            else:
                if current_run > 0:
                    total += current_run * (current_run + 1) // 2
                    current_run = 0
        if current_run > 0:
            total += current_run * (current_run + 1) // 2
        
        n = len(s)
        max_z = 199
        for z in range(1, max_z + 1):
            k = len(pos)
            if k < z:
                continue
            for i in range(k - z + 1):
                first_zero = pos[i]
                last_zero = pos[i + z - 1]
                prev_zero = pos[i - 1] if i > 0 else -1
                if i + z < k:
                    next_zero = pos[i + z]
                else:
                    next_zero = n
                s_min = prev_zero + 1
                s_max = first_zero
                e_min = last_zero
                e_max = next_zero - 1
                if s_min > s_max or e_min > e_max:
                    continue
                l_min = z * (z + 1)
                t = l_min - 1
                c = e_min
                d = e_max
                # Compute sum_part1
                s_high1 = min(s_max, c - t)
                sum_part1 = 0
                if s_high1 >= s_min:
                    count1 = s_high1 - s_min + 1
                    if c <= d:
                        sum_part1 = count1 * (d - c + 1)
                # Compute sum_part2
                s_low2 = max(s_min, c - t + 1)
                s_high2 = min(s_max, d - t)
                sum_part2 = 0
                if s_high2 >= s_low2:
                    C = d - t + 1
                    A = s_low2
                    B = s_high2
                    n_terms = B - A + 1
                    sum_part2 = C * n_terms - (A + B) * n_terms // 2
                total += sum_part1 + sum_part2
        return total