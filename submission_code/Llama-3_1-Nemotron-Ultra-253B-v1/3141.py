from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        def min_subarray_length(arr, target_sum):
            left = 0
            current_sum = 0
            min_len = float('inf')
            for right in range(len(arr)):
                current_sum += arr[right]
                while current_sum > target_sum and left <= right:
                    current_sum -= arr[left]
                    left += 1
                if current_sum == target_sum:
                    min_len = min(min_len, right - left + 1)
            return min_len if min_len != float('inf') else -1
        
        S = sum(nums)
        if S == 0:
            return -1  # As per constraints, nums[i] >= 1, so S cannot be 0
        
        if target >= S:
            k = target // S
            remainder = target % S
            if remainder == 0:
                return k * len(nums)
            else:
                m = min_subarray_length(nums, remainder)
                return k * len(nums) + m if m != -1 else -1
        else:
            concatenated = nums + nums
            m = min_subarray_length(concatenated, target)
            return m if m != -1 else -1