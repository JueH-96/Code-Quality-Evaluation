class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        
        # Calculate cost for each subarray of size x
        costs = []
        for i in range(n - x + 1):
            subarray = nums[i:i + x]
            # Sort to find median
            sorted_sub = sorted(subarray)
            # For minimum operations, change all elements to median
            if x % 2 == 1:
                median = sorted_sub[x // 2]
            else:
                # For even size, either middle element works
                median = sorted_sub[x // 2 - 1]
            
            cost = sum(abs(num - median) for num in subarray)
            costs.append((i, cost))
        
        # DP to find k non-overlapping subarrays with minimum cost
        # dp[i][j] = minimum cost to select j subarrays from first i positions
        m = len(costs)
        INF = float('inf')
        
        # dp[i][j] represents minimum cost using subarrays ending at or before position i
        # to form j non-overlapping subarrays
        dp = [[INF] * (k + 1) for _ in range(m + 1)]
        
        # Base case: 0 subarrays selected
        for i in range(m + 1):
            dp[i][0] = 0
        
        for i in range(1, m + 1):
            start_pos, cost = costs[i - 1]
            
            for j in range(k + 1):
                # Option 1: Don't use current subarray
                dp[i][j] = dp[i - 1][j]
                
                # Option 2: Use current subarray if we can
                if j > 0:
                    # Find the last valid position before current subarray starts
                    prev_end = start_pos - 1
                    # Find how many subarrays we can consider that end before current starts
                    valid_prev = 0
                    for p in range(i):
                        if costs[p][0] + x - 1 < start_pos:
                            valid_prev = p + 1
                    
                    if valid_prev >= j - 1:
                        dp[i][j] = min(dp[i][j], dp[valid_prev][j - 1] + cost)
        
        return dp[m][k]