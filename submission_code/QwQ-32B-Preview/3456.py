from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        # Group nums into runs of identical elements
        groups = []
        current = nums[0]
        count = 1
        for num in nums[1:]:
            if num == current:
                count += 1
            else:
                groups.append((current, count))
                current = num
                count = 1
        groups.append((current, count))
        
        # Get the lengths of the runs
        group_lengths = [length for _, length in groups]
        num_groups = len(groups)
        
        # Initialize DP table
        dp = [[0] * (k + 1) for _ in range(num_groups + 1)]
        
        # Fill DP table
        for i in range(1, num_groups + 1):
            for j in range(k + 1):
                # Option 1: do not select the i-th run
                dp[i][j] = dp[i - 1][j]
                # Option 2: select the i-th run
                if i == 1:
                    dp[i][j] = max(dp[i][j], group_lengths[0])
                else:
                    prev_val, _ = groups[i - 2]
                    curr_val, length = groups[i - 1]
                    if curr_val == prev_val:
                        dp[i][j] = max(dp[i][j], dp[i - 1][j] + length)
                    else:
                        if j >= 1:
                            dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + length)
        
        # Find the maximum length from dp table
        max_length = 0
        for i in range(num_groups + 1):
            for j in range(k + 1):
                max_length = max(max_length, dp[i][j])
        return max_length