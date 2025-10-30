from collections import defaultdict, Counter
from typing import List

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 7:
            return 0
        
        # Precompute prefix frequencies
        prefix_freq = [Counter() for _ in range(n + 1)]
        for i in range(1, n + 1):
            prefix_freq[i] = prefix_freq[i - 1].copy()
            prefix_freq[i][nums[i - 1]] += 1
        
        # Precompute suffix frequencies
        suffix_counts = [defaultdict(int) for _ in range(n + 2)]  # suffix_counts[i] is counts from i to n-1
        for i in range(n - 1, -1, -1):
            suffix_counts[i] = defaultdict(int, suffix_counts[i + 1])
            suffix_counts[i][nums[i]] += 1
        
        result = 0
        
        for q in range(n):
            for r in range(q + 2, n):
                # p must be <= q - 2
                if q == 0:
                    # q-1 = -1, no p's
                    continue
                p_freq = prefix_freq[q - 1]
                current = 0
                for a in p_freq:
                    count_p = p_freq[a]
                    K_p = a * nums[r]
                    if K_p % nums[q] != 0:
                        continue
                    required_s = K_p // nums[q]
                    # s must be >= r + 2
                    s_count = suffix_counts[r + 2].get(required_s, 0)
                    current += count_p * s_count
                result += current
        
        return result