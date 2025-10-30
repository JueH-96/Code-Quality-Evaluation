from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n # Initialize result array of size n
        
        for i in range(n):
            # Calculate the landing index for the current index i.
            # The movement is determined by nums[i]: positive moves right, negative moves left.
            # The circular property is handled by the modulo operator % n.
            # For positive nums[i], (i + nums[i]) % n gives the correct rightward wrap-around index.
            # For negative nums[i], (i + nums[i]) is equivalent to (i - abs(nums[i])).
            # Python's modulo operator handles negative numbers correctly for circular arrays:
            # e.g., (-1) % 5 = 4, (1 - 2) % 4 = -1 % 4 = 3.
            # The landing index j is calculated as (i + nums[i]) % n.
            landing_index = (i + nums[i]) % n
            
            # According to the examples, result[i] should be the value from the original
            # nums array at the calculated landing index.
            # The problem states "Set result[i] to the value of the index where you land."
            # The examples demonstrate this means nums[landing_index].
            # This approach also correctly handles the case where nums[i] == 0,
            # as landing_index would be (i + 0) % n = i, and result[i] = nums[i], which is 0.
            result[i] = nums[landing_index]
            
        return result