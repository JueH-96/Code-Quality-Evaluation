from typing import List

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        count = 0
        while True:
            # Check if the array is non-decreasing
            is_ok = True
            for i in range(len(nums) - 1):
                if nums[i] > nums[i + 1]:
                    is_ok = False
                    break
            if is_ok:
                return count
            
            # Find the leftmost pair with the minimal sum
            min_sum = float('inf')
            best_i = 0
            for i in range(len(nums) - 1):
                current_sum = nums[i] + nums[i + 1]
                if current_sum < min_sum:
                    min_sum = current_sum
                    best_i = i
            
            # Merge the pair
            new_val = nums[best_i] + nums[best_i + 1]
            nums = nums[:best_i] + [new_val] + nums[best_i + 2:]
            count += 1