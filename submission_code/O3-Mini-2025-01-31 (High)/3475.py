from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # count will keep track of the number of operations performed
        count = 0
        n = len(nums)
        
        # We only need to consider starting positions such that the window of 3 lies entirely in the array.
        # That is, for i in [0, n - 3].
        for i in range(n - 2):
            if nums[i] == 0:
                # If nums[i] is 0, we flip the 3 consecutive elements.
                nums[i]   = 1 - nums[i]
                nums[i+1] = 1 - nums[i+1]
                nums[i+2] = 1 - nums[i+2]
                count += 1
        
        # After applying the greedy flipping, check if the entire array is transformed to all 1s.
        if all(num == 1 for num in nums):
            return count
        else:
            return -1