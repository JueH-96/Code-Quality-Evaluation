from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i-1] + nums[i-1]
        
        max_len = [0] * (n + 1)
        # dp_list stores tuples of (key, length, sum_val)
        dp_list = [(0, 0, 0)]  # initial entry for prefix_sum[0]
        
        for i in range(1, n + 1):
            current_prefix = prefix_sum[i]
            left, right = 0, len(dp_list) - 1
            best_idx = -1
            while left <= right:
                mid = (left + right) // 2
                if dp_list[mid][0] <= current_prefix:
                    best_idx = mid
                    left = mid + 1
                else:
                    right = mid - 1
            # best_idx is the index of the best entry
            key_j, len_j, sum_j = dp_list[best_idx]
            prefix_j = key_j - sum_j
            sum_i = current_prefix - prefix_j
            len_i = len_j + 1
            max_len[i] = len_i
            key_i = sum_i + current_prefix
            # Maintain the dp_list
            while dp_list and dp_list[-1][1] <= len_i and dp_list[-1][0] >= key_i:
                dp_list.pop()
            dp_list.append((key_i, len_i, sum_i))
        
        return max_len[n]