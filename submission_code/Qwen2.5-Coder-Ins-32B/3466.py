from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        def count_with_max_and(max_and):
            count = 0
            current_and = (1 << 32) - 1
            left = 0
            for right in range(len(nums)):
                current_and &= nums[right]
                while left <= right and current_and <= max_and:
                    if current_and == max_and:
                        count += left + 1
                    current_and |= nums[left]
                    left += 1
            return count
        
        return count_with_max_and(k) - count_with_max_and(k - 1)