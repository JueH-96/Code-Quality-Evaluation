class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        # We need to choose indices i0, i1, i2, i3 from b such that the score
        # score = a[0] * b[i0] + a[1] * b[i1] + a[2] * b[i2] + a[3] * b[i3]
        # is maximized.
        #
        # We can choose such indices using DP. Since a has fixed length 4,
        # we maintain dp[k] = maximum score achievable by selecting k elements
        # from b in order (i.e. with valid indices order).
        #
        # Initially, dp[0] = 0 (score of selecting no elements) and 
        # dp[1..4] = -infinity (impossible states initially).
        #
        # For each number in b, update dp from backwards k=4 down to 1:
        # dp[k] = max(dp[k], dp[k-1] + a[k-1] * current_b)
        #
        # The final answer is dp[4].
        
        # Instead of float('-inf') we use a sufficiently small number,
        # but in python, we may directly use float('-inf').
        dp = [0, float('-inf'), float('-inf'), float('-inf'), float('-inf')]
        
        for x in b:
            # update the dp in reverse order to ensure correctness
            for k in range(4, 0, -1):
                # We are computing candidate score if we select current x
                dp[k] = max(dp[k], dp[k-1] + a[k-1] * x)
        
        return dp[4]