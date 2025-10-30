from typing import List
from collections import Counter

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 4:
            return 0
        
        # Build prefix array where prefix[i] is the counter for nums[0..i-1]
        prefix = [Counter()]
        current = Counter()
        for i in range(n):
            current[nums[i]] += 1
            prefix.append(current.copy())
        
        # Build suffix array where suffix[i] is the counter for nums[i..n-1]
        suffix = [Counter() for _ in range(n + 1)]
        current = Counter()
        suffix[n] = current.copy()
        for i in range(n - 1, -1, -1):
            current[nums[i]] += 1
            suffix[i] = current.copy()
        
        total = 0
        
        for q in range(n):
            for r in range(q + 2, n):
                b = nums[q]
                c = nums[r]
                # Check if there are elements after r+1
                if r + 2 > n - 1:
                    continue
                d_counts = suffix[r + 2]
                for d in d_counts:
                    if (b * d) % c != 0:
                        continue
                    a = (b * d) // c
                    if q - 1 >= 0:
                        a_count = prefix[q - 1].get(a, 0)
                    else:
                        a_count = 0
                    total += a_count * d_counts[d]
        
        return total