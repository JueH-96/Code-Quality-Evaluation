class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        dp = [-float('inf')] * (target + 1)
        dp[0] = 0  # Base case: sum 0 with 0 elements
        
        for num in nums:
            # Iterate from target down to 0 to avoid reusing the same element
            for s in range(target, -1, -1):
                if dp[s] != -float('inf'):
                    new_sum = s + num
                    if new_sum <= target:
                        # Update the maximum length for new_sum
                        if dp[new_sum] < dp[s] + 1:
                            dp[new_sum] = dp[s] + 1
        
        return dp[target] if dp[target] != -float('inf') else -1