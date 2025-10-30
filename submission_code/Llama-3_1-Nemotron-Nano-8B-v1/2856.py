from typing import List
from collections import defaultdict

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        def at_most_k(k: int) -> int:
            count = 0
            left = 0
            freq = defaultdict(int)
            distinct = 0
            for right in range(len(nums)):
                num = nums[right]
                if freq[num] == 0:
                    distinct += 1
                freq[num] += 1
                while distinct > k:
                    left_num = nums[left]
                    freq[left_num] -= 1
                    if freq[left_num] == 0:
                        distinct -= 1
                    left += 1
                count += right - left + 1
            return count
        
        k_total = len(set(nums))
        return at_most_k(k_total) - at_most_k(k_total - 1)