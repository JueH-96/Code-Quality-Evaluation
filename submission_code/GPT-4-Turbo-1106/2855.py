class Solution:
    def maximumJumps(self, nums: list[int], target: int) -> int:
        n = len(nums)
        dp = [0] + [-1] * (n - 1)  # Initialize dp array with -1 (unreachable)
        
        for i in range(n):
            if dp[i] == -1:
                continue  # Skip if current index is unreachable
            for j in range(i + 1, n):
                if -target <= nums[j] - nums[i] <= target:
                    dp[j] = max(dp[j], dp[i] + 1)
        
        return dp[-1]

# Example usage:
# sol = Solution()
# print(sol.maximumJumps([1,3,6,4,1,2], 2))  # Output: 3
# print(sol.maximumJumps([1,3,6,4,1,2], 3))  # Output: 5
# print(sol.maximumJumps([1,3,6,4,1,2], 0))  # Output: -1