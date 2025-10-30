import math
from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp_prefix[i][m] stores the set of possible OR values for choosing exactly m elements from the first i elements (indices 0 to i-1)
        dp_prefix = [[set() for _ in range(k)] for _ in range(n + 1)]
        dp_prefix[0][0] = {0}
        
        for i in range(1, n + 1):
            for m in range(k):
                # Case where we do not pick nums[i-1]
                dp_prefix[i][m] = set(dp_prefix[i - 1][m])
                # Case where we pick nums[i-1], if m >= 1
                if m >= 1:
                    for mask in dp_prefix[i - 1][m - 1]:
                        dp_prefix[i][m].add(mask | nums[i - 1])
        
        # dp_suffix[i][m] stores the set of possible OR values for choosing exactly m elements from nums[i] to nums[n-1]
        dp_suffix = [[set() for _ in range(k + 1)] for _ in range(n + 1)]
        dp_suffix[n][0] = {0}
        
        for i in range(n - 1, -1, -1):
            for m in range(k + 1):
                # Case where we do not pick nums[i]
                dp_suffix[i][m] = set(dp_suffix[i + 1][m])
                # Case where we pick nums[i], if m >= 1
                if m >= 1:
                    for mask in dp_suffix[i + 1][m - 1]:
                        dp_suffix[i][m].add(mask | nums[i])
        
        # Now iterate over all possible s and compute the maximum XOR value
        ans = 0
        for s in range(k - 1, n - k):
            for mask_left_partial in dp_prefix[s][k - 1]:
                or_left = mask_left_partial | nums[s]
                for or_right in dp_suffix[s + 1][k]:
                    current_xor = or_left ^ or_right
                    if current_xor > ans:
                        ans = current_xor
        
        return ans