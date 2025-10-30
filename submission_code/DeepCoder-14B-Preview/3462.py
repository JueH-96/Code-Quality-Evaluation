from collections import defaultdict
import bisect

class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Compute prefix sum of zeros
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + (1 if s[i - 1] == '0' else 0)
        
        # Build prefix_map: key is prefix value, value is list of indices (i)
        prefix_map = defaultdict(list)
        for idx, val in enumerate(prefix):
            prefix_map[val].append(idx)
        
        # Compute max_z
        max_z = 0
        while (max_z + 1) * (max_z + 2) <= n:
            max_z += 1
        
        count = 0
        
        for j in range(n):
            current_prefix = prefix[j + 1]
            for z in range(0, max_z + 1):
                target = current_prefix - z
                if target not in prefix_map:
                    continue
                i_list = prefix_map[target]
                # Find the rightmost index where i <= j
                idx = bisect.bisect_right(i_list, j)
                for i in i_list[:idx]:
                    length = j - i + 1
                    if length >= z * (z + 1):
                        count += 1
        
        return count