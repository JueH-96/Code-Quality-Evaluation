from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if n == 0:
            return -1
        S = sum(nums)
        doubled = nums + nums
        
        def find_min_subarray(arr, target_sum):
            if target_sum < 0:
                return -1
            left = 0
            current_sum = 0
            min_len = float('inf')
            for right in range(len(arr)):
                current_sum += arr[right]
                while current_sum > target_sum and left <= right:
                    current_sum -= arr[left]
                    left += 1
                if current_sum == target_sum:
                    window_len = right - left + 1
                    if window_len < min_len:
                        min_len = window_len
            return min_len if min_len != float('inf') else -1
        
        min_short = find_min_subarray(doubled, target)
        possible_ans = float('inf')
        
        if min_short != -1:
            possible_ans = min(possible_ans, min_short)
        
        if S != 0:
            k_total = target // S
            rem = target % S
            
            if rem == 0 and k_total > 0:
                possible_ans = min(possible_ans, k_total * n)
            
            if k_total >= 1:
                min_rem = find_min_subarray(doubled, rem)
                if min_rem != -1:
                    possible_ans = min(possible_ans, k_total * n + min_rem)
                
                rem_plus = rem + S
                min_rem_plus = find_min_subarray(doubled, rem_plus)
                if min_rem_plus != -1:
                    possible_ans = min(possible_ans, (k_total - 1) * n + min_rem_plus)
        
        return possible_ans if possible_ans != float('inf') else -1