from bisect import bisect_right
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        if not nums:
            return []
        
        max_num = max(nums)
        freq = [0] * (max_num + 1)
        for num in nums:
            freq[num] += 1
        
        max_d = max_num
        cnt = [0] * (max_d + 1)
        
        for d in range(max_d, 0, -1):
            sum_multiples = 0
            m = d
            while m <= max_d:
                sum_multiples += freq[m]
                m += d
            total = sum_multiples * (sum_multiples - 1) // 2
            
            m = 2 * d
            while m <= max_d:
                total -= cnt[m]
                m += d
            cnt[d] = total
        
        d_list = []
        for d in range(1, max_d + 1):
            if cnt[d] > 0:
                d_list.append(d)
        d_list.sort()
        
        prefix = [0]
        for d in d_list:
            prefix.append(prefix[-1] + cnt[d])
        
        res = []
        for q in queries:
            idx = bisect_right(prefix, q)
            res.append(d_list[idx - 1])
        
        return res