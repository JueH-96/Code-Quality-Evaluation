from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_sum = 0

        # Iterate through each element in the array
        for i in range(n):
            current_num = nums[i]
            
            # Determine if the left-side condition (relative to i - k) is met.
            # It's met if:
            # 1. The left index i - k does not exist (out of bounds), OR
            # 2. The left index exists AND current_num is strictly greater than nums[i - k].
            is_good_left = False
            if i - k < 0:
                is_good_left = True
            else:
                if current_num > nums[i - k]:
                    is_good_left = True
            
            # Determine if the right-side condition (relative to i + k) is met.
            # It's met if:
            # 1. The right index i + k does not exist (out of bounds), OR
            # 2. The right index exists AND current_num is strictly greater than nums[i + k].
            is_good_right = False
            if i + k >= n:
                is_good_right = True
            else:
                if current_num > nums[i + k]:
                    is_good_right = True
            
            # An element nums[i] is considered good if both its left and right conditions are met.
            # If a corresponding index does not exist, its condition is vacuously true.
            if is_good_left and is_good_right:
                total_sum += current_num
                
        return total_sum