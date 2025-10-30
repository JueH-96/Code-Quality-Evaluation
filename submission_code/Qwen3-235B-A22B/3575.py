from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Initialize prefix DP table
        prefix_dp = [[set() for _ in range(k+1)] for _ in range(n+1)]
        prefix_dp[0][0].add(0)
        
        for i in range(1, n+1):
            current_num = nums[i-1]
            for j in range(k+1):
                if j == 0:
                    prefix_dp[i][j].add(0)
                else:
                    # Copy subsets that don't include current_num
                    prefix_dp[i][j] = prefix_dp[i-1][j].copy()
                    # Add subsets that include current_num
                    for prev in prefix_dp[i-1][j-1]:
                        new_or = prev | current_num
                        prefix_dp[i][j].add(new_or)
        
        # Initialize suffix DP table
        suffix_dp = [[set() for _ in range(k+1)] for _ in range(n+2)]
        suffix_dp[n][0].add(0)
        
        for i in range(n-1, -1, -1):
            current_num = nums[i]
            for j in range(k+1):
                if j == 0:
                    suffix_dp[i][0].add(0)
                else:
                    # Copy subsets that don't include current_num
                    suffix_dp[i][j] = suffix_dp[i+1][j].copy()
                    # Add subsets that include current_num
                    if suffix_dp[i+1][j-1]:
                        for prev in suffix_dp[i+1][j-1]:
                            new_or = prev | current_num
                            suffix_dp[i][j].add(new_or)
        
        max_xor = 0
        # Iterate all possible m values
        max_m = n - k
        for m in range(k, max_m + 1):
            pre_set = prefix_dp[m][k]
            suf_set = suffix_dp[m][k]
            if not pre_set or not suf_set:
                continue
            for a in pre_set:
                for b in suf_set:
                    current = a ^ b
                    if current > max_xor:
                        max_xor = current
        return max_xor