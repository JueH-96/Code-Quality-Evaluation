from typing import List
import collections

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        # Total number of distinct elements in the whole array.
        totalDistinct = len(set(nums))
        
        # Helper function: count subarrays with at most x distinct elements.
        def atMostDistinct(x: int) -> int:
            count = 0
            left = 0
            freq = collections.defaultdict(int)
            distinct = 0
            
            for right, num in enumerate(nums):
                freq[num] += 1
                if freq[num] == 1:
                    distinct += 1
                # Shrink the window until the distinct count is <= x.
                while distinct > x:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        distinct -= 1
                    left += 1
                count += (right - left + 1)
            return count

        # The number of subarrays with exactly totalDistinct distinct numbers
        # equals: atMostDistinct(totalDistinct) - atMostDistinct(totalDistinct - 1)
        return atMostDistinct(totalDistinct) - atMostDistinct(totalDistinct - 1)