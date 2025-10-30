from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_sum_of_good_numbers = 0

        for i in range(n):
            current_num = nums[i]
            is_current_element_good = True  # Assume good until proven otherwise

            # Check condition related to the left neighbor nums[i-k]
            # An element nums[i] is considered "good" with respect to the left side if:
            # 1. The left neighbor at i-k does not exist (i.e., i-k < 0), OR
            # 2. nums[i] is strictly greater than nums[i-k].
            # So, nums[i] is NOT "good" w.r.t. left if (left neighbor EXISTS) AND (nums[i] <= nums[i-k]).
            
            left_neighbor_index = i - k
            if left_neighbor_index >= 0:  # If left neighbor exists
                if current_num <= nums[left_neighbor_index]:  # If not strictly greater than left neighbor
                    is_current_element_good = False
            
            # Check condition related to the right neighbor nums[i+k]
            # This check is only performed if the element is still potentially good after the left check.
            # An element nums[i] is considered "good" with respect to the right side if:
            # 1. The right neighbor at i+k does not exist (i.e., i+k >= n), OR
            # 2. nums[i] is strictly greater than nums[i+k].
            # So, nums[i] is NOT "good" w.r.t. right if (right neighbor EXISTS) AND (nums[i] <= nums[i+k]).

            if is_current_element_good:  # Optimization: if already False due to left check, no need for right check
                right_neighbor_index = i + k
                if right_neighbor_index < n:  # If right neighbor exists
                    if current_num <= nums[right_neighbor_index]:  # If not strictly greater than right neighbor
                        is_current_element_good = False
            
            if is_current_element_good:
                total_sum_of_good_numbers += current_num
                
        return total_sum_of_good_numbers