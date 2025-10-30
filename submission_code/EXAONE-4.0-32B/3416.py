from collections import Counter
from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        total_pairs = n * (n - 1) // 2
        d = len(str(nums[0]))
        
        counters = [Counter() for _ in range(d)]
        
        for num in nums:
            s = str(num)
            for idx, char in enumerate(s):
                counters[idx][char] += 1
                
        ans = 0
        for i in range(d):
            same_pairs = 0
            for count in counters[i].values():
                same_pairs += count * (count - 1) // 2
            ans += (total_pairs - same_pairs)
            
        return ans