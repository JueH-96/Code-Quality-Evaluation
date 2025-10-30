from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        # Compute prefix sums: ps[i] = sum of nums[0..i-1]
        ps = [0] * (n + 1)
        for i in range(1, n + 1):
            ps[i] = ps[i - 1] + nums[i - 1]
        
        min_pos_sum = float('inf')
        # For each end index j, consider all valid lengths
        for j in range(1, n + 1):
            # length k in [l, r], start index i = j - k
            for k in range(l, r + 1):
                i = j - k
                if i < 0:
                    continue
                curr_sum = ps[j] - ps[i]
                if curr_sum > 0 and curr_sum < min_pos_sum:
                    min_pos_sum = curr_sum
        
        return min_pos_sum if min_pos_sum != float('inf') else -1