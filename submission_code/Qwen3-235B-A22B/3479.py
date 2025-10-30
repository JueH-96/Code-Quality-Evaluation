import bisect
from collections import defaultdict

class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        prefix_zeros = [0] * (n + 1)
        for i in range(n):
            prefix_zeros[i+1] = prefix_zeros[i] + (1 if s[i] == '0' else 0)
        
        zero_positions = defaultdict(list)
        for i, val in enumerate(prefix_zeros):
            zero_positions[val].append(i)
        
        count = 0
        for j in range(n):
            j_plus_1 = j + 1
            for z in range(0, 201):
                required_length = z * z + z
                if required_length > j_plus_1:
                    continue
                upper_i = j_plus_1 - required_length
                if upper_i > j:
                    upper_i = j
                if upper_i < 0:
                    continue
                target = prefix_zeros[j_plus_1] - z
                if target < 0:
                    continue
                pos_list = zero_positions.get(target, [])
                if not pos_list:
                    continue
                c = bisect.bisect_right(pos_list, upper_i)
                count += c
        return count