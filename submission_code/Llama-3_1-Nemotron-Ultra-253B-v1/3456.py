from collections import defaultdict
from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        dp = defaultdict(int)
        for num in nums:
            temp = defaultdict(int)
            # Process existing states to generate new states by including current num
            for (prev_val, c), length in list(dp.items()):
                new_c = c + (0 if prev_val == num else 1)
                if new_c <= k:
                    temp[(num, new_c)] = max(temp[(num, new_c)], length + 1)
            # Add the option to start a new subsequence with current num
            temp[(num, 0)] = max(temp[(num, 0)], 1)
            # Merge temp into dp, keeping the maximum length for each state
            for key in temp:
                dp[key] = max(dp[key], temp[key])
        return max(dp.values()) if dp else 0