class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        # Initialize a 2D list to store the minimum cost for each partition
        dp = [[float('inf')] * n for _ in range(4)]
        
        # Base case: for 1 subarray, the cost is just the first element
        for i in range(n):
            dp[1][i] = nums[0]
        
        # Fill the dp table for 2 and 3 subarrays
        for k in range(2, 4):  # k = 2 and 3
            for i in range(k - 1, n):  # i must be at least k-1
                for j in range(k - 2, i):  # j must be at least k-2
                    dp[k][i] = min(dp[k][i], dp[k - 1][j] + nums[j + 1])
        
        # The answer is the minimum cost for 3 subarrays
        return min(dp[3][i] for i in range(2, n))