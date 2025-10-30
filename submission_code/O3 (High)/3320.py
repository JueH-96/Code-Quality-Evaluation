from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        """
        Consecutively remove the first two elements as long as the sum of every
        removed pair equals the sum of the first pair.  The answer is therefore
        the length of the longest prefix (in pair units) whose pair‐sums are
        all identical to the first pair’s sum.
        """
        # The first operation is always possible (array length ≥ 2).
        target_sum = nums[0] + nums[1]
        operations = 1                      # first pair
        
        # Examine each subsequent pair (indices 2–3, 4–5, …).
        for i in range(2, len(nums) - 1, 2):
            if nums[i] + nums[i + 1] == target_sum:
                operations += 1            # still same score, keep going
            else:
                break                      # score changed → stop
            
        return operations