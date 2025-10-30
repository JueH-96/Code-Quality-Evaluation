from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        # If there are fewer than 2 elements, no operation can be done.
        if len(nums) < 2:
            return 0
        
        # The required score S is determined by the first operation.
        S = nums[0] + nums[1]
        ops = 0
        
        # Greedily perform operations on the prefix as long as each pair sums to S.
        # Each operation removes two elements, so we step by 2.
        for i in range(0, len(nums) - 1, 2):
            if nums[i] + nums[i + 1] == S:
                ops += 1
            else:
                # once a mismatch occurs, we cannot continue
                break
        
        return ops