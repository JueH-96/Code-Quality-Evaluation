class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # dp[i][j] = XOR score of subarray nums[i..j]
        dp = [[0] * n for _ in range(n)]
        
        # Base case: single elements
        for i in range(n):
            dp[i][i] = nums[i]
        
        # Fill dp table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = dp[i][j-1] ^ dp[i+1][j]
        
        # max_dp[i][j] = maximum XOR score of any subarray within nums[i..j]
        max_dp = [[0] * n for _ in range(n)]
        
        # Base case: single elements
        for i in range(n):
            max_dp[i][i] = nums[i]
        
        # Fill max_dp table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                max_dp[i][j] = max(max_dp[i][j-1], max_dp[i+1][j], dp[i][j])
        
        # Answer queries
        answer = []
        for l, r in queries:
            answer.append(max_dp[l][r])
        
        return answer