class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        # dp[i][k] = minimum cost using some subset of the first i walls
        #            to be able to paint at least k walls in total.
        # A paid wall i lets us paint 1 (itself) + time[i] walls in total.
        
        # Initialize dp array with a large sentinel (infinity).
        INF = 10**15
        dp = [[INF]*(n+1) for _ in range(n+1)]
        
        # Base case: dp[0][0] = 0 (cost 0 to cover 0 walls when we pick none)
        dp[0][0] = 0
        
        for i in range(1, n+1):
            # Paid painter's cost and time for wall i-1
            c = cost[i-1]
            t = time[i-1]
            for k in range(n+1):
                # Do not pick the i-th wall
                dp[i][k] = min(dp[i][k], dp[i-1][k])
                
                # Pick the i-th wall; coverage gained = t+1
                coverage_gained = t + 1
                prev_k = max(0, k - coverage_gained)
                dp[i][k] = min(dp[i][k], dp[i-1][prev_k] + c)
        
        # Minimum cost to cover all n walls
        return dp[n][n]