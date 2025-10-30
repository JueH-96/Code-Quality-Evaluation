from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # dp[s] will hold the maximum length of a subsequence that sums to s.
        # We initialize with -1 to denote that a sum s is not achievable.
        dp = [-1] * (target + 1)
        dp[0] = 0  # Base case: sum of 0 can be achieved with 0 elements.
        
        # Process each number in the given order.
        for num in nums:
            # update from target down to num to avoid using the same element multiple times.
            for s in range(target, num - 1, -1):
                if dp[s - num] != -1:
                    dp[s] = max(dp[s], dp[s - num] + 1)
        
        return dp[target] if dp[target] != -1 else -1

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.lengthOfLongestSubsequence([1,2,3,4,5], 9))   # Expected output: 3
    print(sol.lengthOfLongestSubsequence([4,1,3,2,1,5], 7)) # Expected output: 4
    print(sol.lengthOfLongestSubsequence([1,1,5,4,5], 3))   # Expected output: -1