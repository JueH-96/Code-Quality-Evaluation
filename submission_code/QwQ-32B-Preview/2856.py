from typing import List
from collections import defaultdict

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        def atMostK(k: int, nums: List[int]) -> int:
            count = 0
            left = 0
            freq = defaultdict(int)
            distinct = 0
            for right in range(len(nums)):
                if freq[nums[right]] == 0:
                    distinct += 1
                freq[nums[right]] += 1
                while distinct > k:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        distinct -= 1
                    left += 1
                count += right - left + 1
            return count
        
        total_distinct = len(set(nums))
        return atMostK(total_distinct, nums) - atMostK(total_distinct - 1, nums)