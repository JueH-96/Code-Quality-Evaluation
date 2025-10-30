from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        # If there are fewer than 2 elements, no operation can be performed.
        if len(nums) < 2:
            return 0
        
        # The "target score" is determined by the first pair we remove.
        target_score = nums[0] + nums[1]
        operations = 0
        i = 0
        
        # We iterate through nums in steps of two as long as their sum matches target_score.
        while i < len(nums) - 1:
            current_sum = nums[i] + nums[i + 1]
            if current_sum == target_score:
                operations += 1
                i += 2  # move past the pair we just removed
            else:
                break  # as soon as a pair doesn't match the target score, we stop
        
        return operations