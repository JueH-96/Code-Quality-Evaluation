import bisect
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        max_num = max(nums)
        cnt = [0] * (max_num + 1)
        for num in nums:
            cnt[num] += 1
        
        freq = [0] * (max_num + 1)
        for d in range(max_num, 0, -1):
            total = 0
            multiple = d
            while multiple <= max_num:
                total += cnt[multiple]
                multiple += d
            pairs = total * (total - 1) // 2
            m = 2 * d
            while m <= max_num:
                pairs -= freq[m]
                m += d
            freq[d] = pairs
        
        sorted_ds = []
        for d in range(1, max_num + 1):
            if freq[d] > 0:
                sorted_ds.append(d)
        
        prefix = []
        current_sum = 0
        for d in sorted_ds:
            current_sum += freq[d]
            prefix.append(current_sum)
        
        ans = []
        for q in queries:
            i = bisect.bisect_right(prefix, q)
            ans.append(sorted_ds[i])
        
        return ans