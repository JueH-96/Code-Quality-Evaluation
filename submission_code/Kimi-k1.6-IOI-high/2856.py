from collections import defaultdict
from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        def atMost(k):
            count = 0
            left = 0
            freq = defaultdict(int)
            for right in range(len(nums)):
                freq[nums[right]] += 1
                while len(freq) > k:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        del freq[nums[left]]
                    left += 1
                count += right - left + 1
            return count
        
        total_distinct = len(set(nums))
        return atMost(total_distinct) - atMost(total_distinct - 1)