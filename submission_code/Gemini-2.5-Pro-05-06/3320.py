from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Constraints: 2 <= nums.length <= 100.
        # This means n is always at least 2.
        # Therefore, nums[0] and nums[1] are guaranteed to exist,
        # and the first operation is always performed.
        
        # The score of the first operation. 
        # All subsequent operations must achieve this exact score.
        target_score = nums[0] + nums[1]
        
        # We've already accounted for the first operation.
        operations_count = 1
        
        # current_index points to the first element of the next potential pair
        # in the original nums array.
        # The first pair used indices 0 and 1. So, the next pair would start at index 2.
        current_index = 2
        
        # Continue as long as there are at least two elements remaining
        # from current_index onwards to form the next pair.
        # This means nums[current_index] and nums[current_index + 1] must both be valid indices.
        while current_index + 1 < n:
            # Calculate the sum of the current pair of elements.
            # These are conceptually the "first two elements" of the remaining list.
            sum_of_current_pair = nums[current_index] + nums[current_index + 1]
            
            if sum_of_current_pair == target_score:
                # The current pair's sum matches the target score.
                # So, we "perform" this operation.
                operations_count += 1
                
                # Advance the index to consider the next pair.
                # We skip the two elements just "used".
                current_index += 2
            else:
                # The current pair's sum does not match the target score.
                # No more operations can be performed that satisfy the condition.
                break
                
        return operations_count