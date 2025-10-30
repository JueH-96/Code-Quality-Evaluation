from typing import List

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        # Helper function to check if the array is non-decreasing.
        def is_non_decreasing(arr: List[int]) -> bool:
            for i in range(1, len(arr)):
                if arr[i] < arr[i-1]:
                    return False
            return True

        operations = 0
        # Continue until the array becomes non-decreasing.
        while not is_non_decreasing(nums):
            # When there's only one element, it is by definition non-decreasing.
            if len(nums) < 2:
                break
            
            # Find the adjacent pair with the minimum sum. In case of a tie, choose the leftmost.
            min_sum = float('inf')
            min_index = 0
            for i in range(len(nums) - 1):
                pair_sum = nums[i] + nums[i+1]
                if pair_sum < min_sum:
                    min_sum = pair_sum
                    min_index = i
            
            # Replace the two numbers at min_index and min_index+1 with their sum.
            nums = nums[:min_index] + [min_sum] + nums[min_index+2:]
            operations += 1
        
        return operations