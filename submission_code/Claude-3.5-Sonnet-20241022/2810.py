class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # dp[i][j] represents min cost to get chocolate type i after j rotations
        dp = [[float('inf')] * n for _ in range(n)]
        
        # Fill initial costs
        for i in range(n):
            for j in range(n):
                # After j rotations, chocolate at position i will be at (i-j+n)%n
                pos = (i-j+n)%n 
                dp[i][j] = nums[pos]
                
                # Take minimum of current and previous rotation
                if j > 0:
                    dp[i][j] = min(dp[i][j], dp[i][j-1])
        
        ans = float('inf')
        # Try each number of rotations from 0 to n-1
        for rot in range(n):
            curr = rot * x  # Cost of rotations
            # Add minimum cost for each chocolate type after rot rotations
            for i in range(n):
                curr += dp[i][rot]
            ans = min(ans, curr)
            
        return ans