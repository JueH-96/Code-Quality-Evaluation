from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        # Iterate through all possible starting indices a
        for a in range(len(nums) - 2 * k + 1):
            # Check if the first subarray is strictly increasing
            if all(nums[a + i] < nums[a + i + 1] for i in range(k - 1)):
                # Check if the second subarray is strictly increasing
                if all(nums[a + k + i] < nums[a + k + i + 1] for i in range(k - 1)):
                    return True
        return False