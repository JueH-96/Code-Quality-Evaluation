class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        """
        We want to pick exactly 4 indices from b (in strictly increasing order)
        so as to maximize the sum:
            a[0]*b[i0] + a[1]*b[i1] + a[2]*b[i2] + a[3]*b[i3].
        
        A neat way to solve this is via dynamic programming. We let dp[j][k] be
        the maximum possible sum when choosing j elements from the first k elements of b.
        
        However, for large b (up to 10^5), storing a 2D dp array of size 5 x (n+1)
        might be memory-intensive (though still feasible). We can optimize by only
        storing the previous row (dp for j-1) and the current row (dp for j).
        
        The state transition is:
            dp[j][k] = max( dp[j][k-1], dp[j-1][k-1] + a[j-1]*b[k-1] )
        
        Meaning we either skip the k-th b-element, or pick it (adding a[j-1]*b[k-1])
        to the result of dp[j-1][k-1].
        
        Finally, our answer will be dp[4][n], i.e. picking 4 elements out of 
        the full length of b.
        """
        
        n = len(b)
        
        # dp0 will correspond to dp[j-1], dp1 to dp[j].
        # dp0[k] = best sum for "j-1 picks" using first k elements of b.
        dp0 = [0] * (n + 1)  # For j=0, picking 0 elements => sum = 0
        
        # We need to do exactly 4 picks, so j goes from 1 to 4.
        for j in range(1, 5):
            dp1 = [float('-inf')] * (n + 1)
            # We can't pick any elements out of zero length, so dp1[0] remains -inf (for j>0).
            for k in range(1, n + 1):
                # Option 1: skip the k-th b element
                skip_val = dp1[k - 1]
                
                # Option 2: pick the k-th b element
                pick_val = dp0[k - 1] + a[j - 1] * b[k - 1]
                
                dp1[k] = max(skip_val, pick_val)
            
            dp0 = dp1  # Move current dp to previous for the next iteration
        
        return dp0[n]