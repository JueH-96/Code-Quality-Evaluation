class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def isPerfectSquare(n):
            root = int(n ** 0.5)
            return root * root == n

        nums.sort()
        dp = [0] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if isPerfectSquare(nums[i] + nums[j]):
                    dp[i] = max(dp[i], dp[j] + nums[i])
        return max(dp + [nums[-1]])