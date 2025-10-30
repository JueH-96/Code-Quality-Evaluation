from typing import List
from collections import defaultdict

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        # Initialize DP and max_len
        dp = [defaultdict(int) for _ in range(k + 1)]
        max_len = [0] * (k + 1)
        
        for x in nums:
            for t in reversed(range(k + 1)):
                if t == 0:
                    # Handle t=0 case: subsequence consists of same elements
                    current_length = dp[t].get(x, 0) + 1
                    if current_length > dp[t][x]:
                        dp[t][x] = current_length
                        if current_length > max_len[t]:
                            max_len[t] = current_length
                else:
                    # For t>0: two possibilities
                    max_prev = max_len[t - 1] if (t - 1) >= 0 else 0
                    current_length = max(max_prev + 1, dp[t].get(x, 0) + 1)
                    if current_length > dp[t][x]:
                        dp[t][x] = current_length
                        if current_length > max_len[t]:
                            max_len[t] = current_length
        # The maximum possible is the maximum value in max_len
        return max(max_len)