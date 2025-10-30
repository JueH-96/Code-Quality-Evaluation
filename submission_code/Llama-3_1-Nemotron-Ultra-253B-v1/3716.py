from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        max_length = 1
        suffix_max_list = []
        for i in range(n):
            dp_i = [0] * 300
            for j in range(i):
                d_ij = abs(nums[i] - nums[j])
                if d_ij >= 300:
                    max_prev = 0
                else:
                    suffix_max_j = suffix_max_list[j]
                    max_prev = suffix_max_j[d_ij] if d_ij < 300 else 0
                new_length = max_prev + 1 if max_prev > 0 else 2
                if new_length > dp_i[d_ij]:
                    dp_i[d_ij] = new_length
            # Compute suffix_max for i
            suffix_max_i = [0] * 300
            current_max = 0
            for d in range(299, -1, -1):
                current_max = max(current_max, dp_i[d])
                suffix_max_i[d] = current_max
            suffix_max_list.append(suffix_max_i)
            if suffix_max_i[0] > max_length:
                max_length = suffix_max_i[0]
        return max_length if max_length >= 2 else 1