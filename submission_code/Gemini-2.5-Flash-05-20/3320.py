from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        # The common score for all operations must be the sum of the first two elements
        # of the *original* array. This is because if the first operation
        # (nums[0] + nums[1]) does not match the chosen common score, then we cannot
        # perform the first operation at all, resulting in 0 operations.
        # By choosing nums[0] + nums[1] as the target score, we are guaranteed to
        # perform at least 1 operation (since nums.length is at least 2).
        
        target_score = nums[0] + nums[1]
        operations_count = 0
        
        # 'i' serves as a pointer to the start of the current conceptual 'first' element.
        # We process elements at indices 'i' and 'i+1'.
        i = 0
        while i + 1 < len(nums):
            # Calculate the sum of the current two elements
            current_sum = nums[i] + nums[i+1]
            
            if current_sum == target_score:
                # If the sum matches the target score, we perform an operation
                operations_count += 1
                # Advance the pointer by 2, effectively 'deleting' the current pair
                i += 2
            else:
                # If the sum does not match, we cannot continue performing operations
                # with this target score. So, we stop.
                break
                
        return operations_count