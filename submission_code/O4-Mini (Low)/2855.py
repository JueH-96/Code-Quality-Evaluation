from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # dp[i] will store the maximum number of jumps to reach index i,
        # or -1 if i is unreachable.
        dp = [-1] * n
        dp[0] = 0
        
        for j in range(1, n):
            # Try jumping to j from any previous i < j
            best = -1
            for i in range(j):
                if dp[i] >= 0 and abs(nums[j] - nums[i]) <= target:
                    best = max(best, dp[i] + 1)
            dp[j] = best
        
        return dp[n - 1]

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maximumJumps([1,3,6,4,1,2], 2))  # Output: 3
    print(sol.maximumJumps([1,3,6,4,1,2], 3))  # Output: 5
    print(sol.maximumJumps([1,3,6,4,1,2], 0))  # Output: -1