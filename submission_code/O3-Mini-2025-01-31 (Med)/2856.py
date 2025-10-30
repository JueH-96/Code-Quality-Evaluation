from collections import defaultdict
from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        # Count the total distinct numbers in the whole array.
        total_distinct = len(set(nums))
        
        # Helper function: count of subarrays with at most k distinct elements.
        def atMostK(k: int) -> int:
            count = defaultdict(int)
            res = 0
            left = 0
            distinct = 0
            # Use two pointers sliding window.
            for right in range(len(nums)):
                # If this number is new in the window, increase the distinct count.
                if count[nums[right]] == 0:
                    distinct += 1
                count[nums[right]] += 1

                # Shrink the window from left until distinct <= k.
                while distinct > k:
                    count[nums[left]] -= 1
                    if count[nums[left]] == 0:
                        distinct -= 1
                    left += 1
                
                # For fixed right, there are (right - left + 1) subarrays ending at right.
                res += (right - left + 1)
            return res
        
        # If we need exactly total_distinct distinct elements in a subarray
        # count_exactly = atMostK(total_distinct) - atMostK(total_distinct - 1)
        return atMostK(total_distinct) - atMostK(total_distinct - 1)