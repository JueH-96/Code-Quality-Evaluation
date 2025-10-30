class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        if x * k > n:
            return 0  # According to constraints, this case won't occur
        
        # Compute the cost array for each possible starting index of a window of size x
        cost = []
        for s in range(n - x + 1):
            window = nums[s:s+x]
            window_sorted = sorted(window)
            m = len(window_sorted) // 2
            median = window_sorted[m]
            total = 0
            for num in window:
                total += abs(num - median)
            cost.append(total)
        
        INF = float('inf')
        # Initialize DP table: dp[i][j] is the minimal cost for first i elements with j windows
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0  # Base case: 0 elements, 0 windows
        
        for i in range(1, n + 1):
            for j in range(k + 1):
                if j == 0:
                    dp[i][j] = dp[i-1][j]
                else:
                    # Option 1: do not take the current element as part of a new window
                    option1 = dp[i-1][j]
                    # Option 2: take the window ending at i (starting at i-x)
                    option2 = INF
                    if i >= x:
                        s_0 = i - x  # starting index of the window (0-based)
                        if s_0 >= 0 and s_0 < len(cost):
                            option2 = dp[s_0][j-1] + cost[s_0]
                    dp[i][j] = min(option1, option2)
        
        return dp[n][k]