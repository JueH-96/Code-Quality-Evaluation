class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        # Count substrings with zero zeros (all 1's)
        z0_count = 0
        current_run = 0
        for c in s:
            if c == '1':
                current_run += 1
            else:
                z0_count += current_run * (current_run + 1) // 2
                current_run = 0
        z0_count += current_run * (current_run + 1) // 2
        
        # Prepare zero indices with padding
        zero_indices = [-1]
        for i, c in enumerate(s):
            if c == '0':
                zero_indices.append(i)
        zero_indices.append(len(s))
        
        total = z0_count
        max_z = int(len(s)**0.5) + 1  # Upper bound for possible z
        
        for z in range(1, max_z + 1):
            L = z * z + z
            if L > len(s):
                continue
            # Iterate over all possible starting zero positions for exactly z zeros
            for i in range(1, len(zero_indices) - z):
                a = zero_indices[i-1] + 1
                b = zero_indices[i]
                c = zero_indices[i + z - 1]
                d = zero_indices[i + z] - 1
                
                s_start = a
                s_end = min(b, d - L + 1)
                if s_start > s_end:
                    continue
                
                s_split = c - L + 1
                # Part 1: s <= s_split
                part1_s_start = s_start
                part1_s_end = min(s_end, s_split)
                if part1_s_start <= part1_s_end:
                    num_part1 = part1_s_end - part1_s_start + 1
                    total_part1 = (d - c + 1) * num_part1
                else:
                    total_part1 = 0
                
                # Part 2: s > s_split
                part2_s_start = max(s_start, s_split + 1)
                part2_s_end = s_end
                if part2_s_start <= part2_s_end:
                    count = part2_s_end - part2_s_start + 1
                    sum_e_part2 = (d - L + 2) * count - (part2_s_start + part2_s_end) * count // 2
                    total_part2 = sum_e_part2
                else:
                    total_part2 = 0
                
                total += total_part1 + total_part2
        
        return total