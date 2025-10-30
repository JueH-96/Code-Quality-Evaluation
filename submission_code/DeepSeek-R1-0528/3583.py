import bisect
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        max_val = max(nums)
        freq = [0] * (max_val + 1)
        for num in nums:
            freq[num] += 1
        
        cnt = [0] * (max_val + 1)
        for d in range(1, max_val + 1):
            total = 0
            j = d
            while j <= max_val:
                total += freq[j]
                j += d
            cnt[d] = total
        
        f = [0] * (max_val + 1)
        for d in range(max_val, 0, -1):
            total_pairs = cnt[d] * (cnt[d] - 1) // 2
            j = 2 * d
            while j <= max_val:
                total_pairs -= f[j]
                j += d
            f[d] = total_pairs
        
        d_list = []
        cumul_list = []
        current_cumul = 0
        for d in range(1, max_val + 1):
            if f[d] > 0:
                current_cumul += f[d]
                d_list.append(d)
                cumul_list.append(current_cumul)
        
        res = []
        for k in queries:
            idx = bisect.bisect_left(cumul_list, k + 1)
            res.append(d_list[idx])
        return res