from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        sum_nums = sum(nums)
        
        def get_min_len(arr, target_val):
            min_len = float('inf')
            current_sum = 0
            start = 0
            for end in range(len(arr)):
                current_sum += arr[end]
                while current_sum > target_val:
                    current_sum -= arr[start]
                    start += 1
                if current_sum == target_val:
                    min_len = min(min_len, end - start + 1)
            if min_len == float('inf'):
                return -1
            return min_len

        if sum_nums == 0:
            return -1 if target != 0 else 1 # Not possible as nums[i] >= 1

        if target < sum_nums:
            extended_nums = nums * 2
            min_len = get_min_len(extended_nums, target)
            return min_len
        else:
            k = target // sum_nums
            rem_target = target % sum_nums
            if rem_target == 0:
                return k * n
            else:
                extended_nums = nums * 2
                min_len_rem = get_min_len(extended_nums, rem_target)
                if min_len_rem == -1:
                    return -1
                return k * n + min_len_rem