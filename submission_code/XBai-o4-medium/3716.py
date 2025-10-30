from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        max_diff = 299  # since the maximum possible difference between 1 and 300 is 299
        n = len(nums)
        # dp[i][d] will store the maximum length of subsequences ending at i with last difference d
        dp = [[0] * (max_diff + 1) for _ in range(n)]
        # max_len[i][d] will store the maximum length for differences >= d for subsequences ending at i
        max_len = [[0] * (max_diff + 1) for _ in range(n)]
        res = 1  # The minimum possible is 1 (a single element)
        
        for i in range(n):
            # Process all previous elements j < i
            for j in range(i):
                current_diff = abs(nums[i] - nums[j])
                # Get the maximum possible length from j's max_len for current_diff
                possible_length = max_len[j][current_diff]
                new_length = possible_length + 1
                if new_length > dp[i][current_diff]:
                    dp[i][current_diff] = new_length
            
            # Compute max_len[i] for this i
            current_max = 1  # Start with the virtual entry (subsequence of length 1)
            for d in range(max_diff, -1, -1):
                if dp[i][d] > 0 and dp[i][d] > current_max:
                    current_max = dp[i][d]
                max_len[i][d] = current_max
            
            # Update the global result with the maximum length ending at this i
            if max_len[i][0] > res:
                res = max_len[i][0]
        
        return res