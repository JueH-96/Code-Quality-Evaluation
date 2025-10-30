from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)
        n = len(nums)
        
        if target == total_sum:
            return n
        elif target < total_sum:
            # Consider up to two repetitions of nums
            two_nums = nums + nums
            min_len = self.min_subarray_length_exactly(two_nums, target)
            return min_len if min_len != -1 else -1
        else:
            # target > total_sum
            k = target // total_sum
            remainder = target - k * total_sum
            if remainder == 0:
                return k * n
            else:
                min_len_remainder = self.min_subarray_length_exactly(nums, remainder)
                if min_len_remainder == -1:
                    return -1
                else:
                    return k * n + min_len_remainder
    
    def min_subarray_length_exactly(self, arr: List[int], target: int) -> int:
        prefix_sum = 0
        min_length = float('inf')
        sum_index = {0: -1}  # To handle subarrays starting from index 0
        
        for i, num in enumerate(arr):
            prefix_sum += num
            if prefix_sum - target in sum_index:
                start_index = sum_index[prefix_sum - target]
                subarray_length = i - start_index
                min_length = min(min_length, subarray_length)
            if prefix_sum not in sum_index:
                sum_index[prefix_sum] = i  # Only store the first occurrence for minimal length
        
        return min_length if min_length != float('inf') else -1