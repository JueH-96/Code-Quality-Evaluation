from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        # Initialize the result with the bitwise OR of all elements in nums
        result = 0
        for num in nums:
            result |= num

        # Perform at most k operations
        for _ in range(k):
            # Find the highest bit that can be cleared
            highest_bit = 1 << 30
            while highest_bit > 0 and (result & highest_bit) == 0:
                highest_bit >>= 1

            if highest_bit == 0:
                break

            # Try to clear the highest bit
            can_clear = False
            for i in range(len(nums) - 1):
                if (nums[i] & highest_bit) and (nums[i + 1] & highest_bit):
                    nums[i] = nums[i] & nums[i + 1]
                    nums.pop(i + 1)
                    can_clear = True
                    break

            if not can_clear:
                break

            # Update the result
            result = 0
            for num in nums:
                result |= num

        return result