from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [{} for _ in range(k + 1)]
        first_num = nums[0]
        dp[0][first_num] = 1
        for i in range(1, n):
            num = nums[i]
            new_dp = []
            for t in range(k + 1):
                new_dp.append(dp[t].copy())
            for t_prev in range(k + 1):
                for val_prev in dp[t_prev]:
                    current_length = dp[t_prev][val_prev]
                    if num == val_prev:
                        new_t = t_prev
                    else:
                        new_t = t_prev + 1
                    if new_t > k:
                        continue
                    if num in new_dp[new_t]:
                        if new_dp[new_t][num] < current_length + 1:
                            new_dp[new_t][num] = current_length + 1
                    else:
                        new_dp[new_t][num] = current_length + 1
            dp = new_dp
        max_len = 0
        for t in range(k + 1):
            if dp[t]:
                current_max = max(dp[t].values())
                max_len = max(max_len, current_max)
        return max_len