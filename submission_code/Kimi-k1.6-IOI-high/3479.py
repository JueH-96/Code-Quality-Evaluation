class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        zeros = [i for i, c in enumerate(s) if c == '0']
        n = len(s)
        total = 0
        
        # Calculate all-ones substrings
        prev = -1
        for i in range(n + 1):
            if i < n and s[i] == '1':
                continue
            else:
                m = i - prev - 1
                total += m * (m + 1) // 2
                prev = i
        
        if not zeros:
            return total
        
        max_k = 200  # Empirical upper bound to cover all possible cases
        
        for k in range(1, max_k + 1):
            required_length = k * k + k
            if required_length > n:
                continue
            num_groups = len(zeros) - k + 1
            if num_groups <= 0:
                continue
            
            for i in range(num_groups):
                # Calculate start range [s_start, s_end]
                if i == 0:
                    s_start = 0
                else:
                    s_start = zeros[i - 1] + 1
                s_end = zeros[i]
                
                # Calculate end range [e_start, e_end]
                e_start = zeros[i + k - 1]
                if i + k < len(zeros):
                    e_end = zeros[i + k] - 1
                else:
                    e_end = n - 1
                
                # Check if start and end ranges are valid
                if s_end < s_start or e_end < e_start:
                    continue
                
                a = max(e_start, required_length + s_start - 1)
                b_val = e_end
                if a > b_val:
                    continue
                
                a_prime = a - required_length + 1
                b_prime = b_val - required_length + 1
                
                # Compute sum_min for e_prime in [a_prime, b_prime]
                s_end_val = s_end
                if a_prime > b_prime:
                    continue  # No valid terms
                if a_prime > s_end_val:
                    sum_min = s_end_val * (b_prime - a_prime + 1)
                elif b_prime < s_end_val:
                    sum_min = (a_prime + b_prime) * (b_prime - a_prime + 1) // 2
                else:
                    part1_end = s_end_val
                    part1_length = part1_end - a_prime + 1
                    sum_part1 = (a_prime + part1_end) * part1_length // 2
                    part2_length = b_prime - part1_end
                    sum_part2 = s_end_val * part2_length
                    sum_min = sum_part1 + sum_part2
                
                num_terms = b_prime - a_prime + 1
                current = sum_min - (s_start - 1) * num_terms
                total += current
        
        return total