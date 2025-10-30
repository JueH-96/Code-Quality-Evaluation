class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        zeros = [i for i, c in enumerate(s) if c == '0']
        m = len(zeros)
        total = 0
        
        # Handle substrings with zero zeros
        prev_zero = -1
        for i in range(m):
            current_zero = zeros[i]
            prev_zero_pos = zeros[i-1] if i > 0 else -1
            start = prev_zero_pos + 1
            end = current_zero - 1
            if start > end:
                continue
            run_length = end - start + 1
            total += run_length * (run_length + 1) // 2
        
        # Handle substrings with exactly z consecutive zeros
        for i in range(m):
            z = i + 1
            if z * (z + 1) > n:
                continue
            group_start = zeros[i]
            group_end = zeros[i + z - 1]
            prev_zero_pos = zeros[i - 1] if i > 0 else -1
            next_zero_pos = zeros[i + z] if (i + z) < m else n
            L = prev_zero_pos + 1 if i > 0 else 0
            R = group_end
            N = next_zero_pos - 1
            
            min_length = R - L + 1
            required = z * z + z
            if min_length >= required:
                count = (R - L + 1) * (N - M + 1)
                total += count
            else:
                start_s = max(L, M - required + 2)
                end_s = min(R, N - required + 1)
                if start_s > end_s:
                    contribution = 0
                else:
                    a = start_s
                    b = end_s
                    count_e = b - a + 1
                    sum_e = a * (a - 1) // 2 - (b * (b - 1) // 2)
                    sum_total = count_e * (required - 1) + sum_e
                    contribution = sum_total
                total += contribution
        
        return total