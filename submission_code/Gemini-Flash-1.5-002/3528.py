class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0
        
        dp = {}
        def solve(index):
            if index == n - 1:
                return 0
            if index in dp:
                return dp[index]
            
            max_score = -1
            for j in range(index + 1, n):
                score = (j - index) * nums[index] + solve(j)
                max_score = max(max_score, score)
            dp[index] = max_score
            return max_score

        return solve(0)