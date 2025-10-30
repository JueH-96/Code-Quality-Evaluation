from typing import List
from collections import defaultdict

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        K = len(set(nums))
        n = len(nums)
        if K == 0:
            return 0  # as per problem constraints, this case won't occur
        
        total_subarrays = n * (n + 1) // 2
        
        def atMost(x):
            count = 0
            left = 0
            freq = defaultdict(int)
            res = 0
            for right in range(n):
                num = nums[right]
                if freq[num] == 0:
                    count += 1
                freq[num] += 1
                
                while count > x:
                    left_num = nums[left]
                    freq[left_num] -= 1
                    if freq[left_num] == 0:
                        count -= 1
                    left += 1
                res += right - left + 1
            return res
        
        return total_subarrays - atMost(K - 1)