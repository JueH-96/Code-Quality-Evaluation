from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        num_ops = n - k + 1
        prefix_sum = [0] * (num_ops + 1)
        
        for i in range(num_ops):
            L = max(0, i - k + 1)
            if L <= i - 1:
                sum_prev = prefix_sum[i] - prefix_sum[L]
            else:
                sum_prev = 0
            required = nums[i] - sum_prev
            if required < 0:
                return False
            prefix_sum[i + 1] = prefix_sum[i] + required
        
        for j in range(n - k + 1, n):
            L_j = max(0, j - k + 1)
            U_j = n - k
            sum_sj = prefix_sum[U_j + 1] - prefix_sum[L_j]
            if sum_sj != nums[j]:
                return False
        
        return True