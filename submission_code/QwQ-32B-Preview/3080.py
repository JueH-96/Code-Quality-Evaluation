from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Step 1: Calculate the bitwise AND of the entire array
        total_and = nums[0]
        for num in nums:
            total_and &= num
        
        # If total_and is not zero, return 1
        if total_and != 0:
            return 1
        
        # Step 2: Find the maximum number of subarrays with bitwise AND zero
        n = len(nums)
        count = 0
        i = 0
        while i < n:
            current_and = nums[i]
            j = i
            # Extend the subarray until the bitwise AND becomes zero
            while current_and != 0 and j < n - 1:
                j += 1
                current_and &= nums[j]
            # If the bitwise AND is zero, increment count and move to the next subarray
            if current_and == 0:
                count += 1
                i = j + 1
            else:
                # If no subarray starting at i can have AND zero, break
                break
        return count