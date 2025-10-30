from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        def find_min_subarray_length(arr: List[int], target_sum: int) -> int:
            min_len = float('inf')
            current_sum = 0
            start = 0
            for end in range(len(arr)):
                current_sum += arr[end]
                while current_sum > target_sum and start <= end:
                    current_sum -= arr[start]
                    start += 1
                if current_sum == target_sum:
                    min_len = min(min_len, end - start + 1)
            return min_len if min_len != float('inf') else -1

        sum_num = sum(nums)
        if sum_num == 0:
            return -1 if target !=0 else 1

        k_max = target // sum_num
        possible_ks = [k_max, k_max + 1]
        if k_max > 0:
            possible_ks.append(k_max -1)
        
        min_len = float('inf')
        
        for k in possible_ks:
            if k <0:
                continue
            target_remaining = target - k * sum_num
            if target_remaining <0:
                continue
            if target_remaining ==0:
                min_len = min(min_len, k * len(nums))
            else:
                sub_len = find_min_subarray_length(nums, target_remaining)
                if sub_len != -1:
                    total_len = k * len(nums) + sub_len
                    min_len = min(min_len, total_len)
        
        return min_len if min_len != float('inf') else -1