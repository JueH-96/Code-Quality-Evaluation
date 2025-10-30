import collections
from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        """
        Calculates the maximum number of operations where each operation consists of
        deleting the first two elements, and all operations have the same sum.
        """
        # Constraints: 2 <= nums.length <= 100, so the first operation is always possible.
        
        # The score of the first operation is determined by the sum of the first two elements.
        # This score becomes the target for all subsequent operations.
        target_score = nums[0] + nums[1]
        
        # We start with one successful operation.
        operations_count = 1
        
        n = len(nums)
        
        # We check the rest of the array in pairs.
        # The loop iterates from index 2, stepping by 2. 
        # The range `range(2, n - 1, 2)` ensures we only access indices `i` and `i + 1`
        # when both are valid. This lets us check pairs `(nums[2], nums[3])`, 
        # `(nums[4], nums[5])`, and so on, as long as they exist.
        for i in range(2, n - 1, 2):
            # If the sum of the current pair matches our target score...
            if nums[i] + nums[i+1] == target_score:
                # ...we increment the operation count.
                operations_count += 1
            else:
                # If it doesn't match, the condition that all operations have the
                # same score is violated. We cannot continue, so we break the loop.
                break
                
        return operations_count