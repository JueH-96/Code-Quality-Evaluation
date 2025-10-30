from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        # There is an inherent order in the operations.
        # The first operation will remove the first two elements and set the target score.
        # Then, while there are at least two elements, we remove the first two.
        # If their sum equals the target score, we count the operation;
        # if not, we stop performing further operations.
        #
        # This procedure yields the maximum number of operations with the same score,
        # because we are not allowed to skip or reorder elements – the operation always
        # takes the first two elements of the current list.
        #
        # Example:
        # nums = [3,2,1,4,5]: target = 3+2 = 5.
        # Next pair is [1,4] with sum=5, so count becomes 2.
        # The remaining element 5 is not enough to form a pair.
        #
        # Example:
        # nums = [3,2,6,1,4]: target = 3+2 = 5.
        # Next pair is [6,1] with sum=7 (not equal to 5), so we stop and answer is 1.
        
        # If there are fewer than 2 elements, no operations can be performed.
        if len(nums) < 2:
            return 0
        
        # Set the target score from the first two elements.
        target = nums[0] + nums[1]
        ops = 0
        # We'll simulate removal of the first two elements repeatedly.
        # Using an index pointer is easier than repeatedly splicing the list.
        i = 0
        # While there are at least 2 elements from index i onward.
        while i + 1 < len(nums):
            # Calculate the sum of the current pair.
            current_sum = nums[i] + nums[i+1]
            if current_sum != target:
                break
            ops += 1
            i += 2  # Move to the next pair.
        
        return ops