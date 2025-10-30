from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        min_sum = float('inf')
        
        for i in range(n):
            max_possible_k = n - i
            if max_possible_k < l:
                continue
            start_k = l
            end_k = min(r, max_possible_k)
            for k in range(start_k, end_k + 1):
                end = i + k
                current_sum = prefix[end] - prefix[i]
                if current_sum > 0 and current_sum < min_sum:
                    min_sum = current_sum
        
        return min_sum if min_sum != float('inf') else -1