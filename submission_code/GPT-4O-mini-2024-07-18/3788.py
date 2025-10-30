from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        seen = set()
        max_sum = 0
        current_sum = 0
        
        for num in nums:
            if num not in seen:
                seen.add(num)
                current_sum += num
            else:
                # If we see a duplicate, we need to reset the current sum
                # and start a new subarray from the last occurrence of the duplicate
                while num in seen:
                    seen.remove(nums[current_sum - len(seen)])
                seen.add(num)
                current_sum += num
            
            max_sum = max(max_sum, current_sum)
        
        return max_sum