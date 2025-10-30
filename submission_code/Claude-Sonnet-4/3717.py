class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        
        # Calculate cost for each possible subarray of size x
        costs = []
        for i in range(n - x + 1):
            subarray = nums[i:i+x]
            subarray.sort()
            median = subarray[x // 2]  # For minimizing operations, use median
            cost = sum(abs(val - median) for val in subarray)
            costs.append((i, cost))
        
        # DP to find minimum cost to select k non-overlapping subarrays
        # dp[i][j] = minimum cost to select j subarrays from first i positions
        m = len(costs)
        dp = [[float('inf')] * (k + 1) for _ in range(m + 1)]
        
        # Base case: 0 subarrays selected = 0 cost
        for i in range(m + 1):
            dp[i][0] = 0
        
        for i in range(1, m + 1):
            pos, cost = costs[i-1]
            
            for j in range(1, min(i + 1, k + 1)):
                # Option 1: Don't select current subarray
                dp[i][j] = dp[i-1][j]
                
                # Option 2: Select current subarray
                # Find the rightmost previous subarray that doesn't overlap
                prev_end = pos - 1
                prev_idx = i - 1
                
                # Find last subarray that ends before current one starts
                while prev_idx > 0 and costs[prev_idx-1][0] + x - 1 > prev_end:
                    prev_idx -= 1
                
                if j == 1:
                    dp[i][j] = min(dp[i][j], cost)
                else:
                    dp[i][j] = min(dp[i][j], dp[prev_idx][j-1] + cost)
        
        return dp[m][k]