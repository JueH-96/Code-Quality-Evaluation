from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        S = sum(nums)
        L = len(nums)
        
        if S > target:
            extended = nums * 2
            min_len = self.helper(extended, target)
            if min_len == float('inf'):
                return -1
            else:
                return min_len
        else:
            k = target // S
            r = target % S
            if r == 0:
                return k * L
            else:
                extended = nums * 2
                len_r = self.helper(extended, r)
                len_r_plus_S = self.helper(extended, r + S)
                option1 = k * L + len_r
                option2 = (k - 1) * L + len_r_plus_S
                res = min(option1, option2)
                if res == float('inf'):
                    return -1
                else:
                    return res
    
    def helper(self, extended, target_sum):
        current_sum = 0
        left = 0
        min_len = float('inf')
        for right in range(len(extended)):
            current_sum += extended[right]
            while current_sum > target_sum and left <= right:
                current_sum -= extended[left]
                left += 1
            if current_sum == target_sum:
                min_len = min(min_len, right - left + 1)
        return min_len if min_len != float('inf') else float('inf')