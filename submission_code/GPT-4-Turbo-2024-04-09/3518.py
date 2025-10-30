class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        # We need to find the maximum possible score using the formula:
        # a[0]*b[i_0] + a[1]*b[i_1] + a[2]*b[i_2] + a[3]*b[i_3] where i_0 < i_1 < i_2 < i_3
        
        # We can use dynamic programming to keep track of the best scores at each stage
        # dp1, dp2, dp3, dp4 will represent the best scores we can achieve using 1, 2, 3, and 4 elements of b respectively
        
        # Initialize dp arrays with very small numbers
        dp1 = float('-inf')
        dp2 = float('-inf')
        dp3 = float('-inf')
        dp4 = float('-inf')
        
        for i in range(n):
            # Update dp4 using the previous value of dp3
            if i >= 3:
                dp4 = max(dp4, dp3 + a[3] * b[i])
            # Update dp3 using the previous value of dp2
            if i >= 2:
                dp3 = max(dp3, dp2 + a[2] * b[i])
            # Update dp2 using the previous value of dp1
            if i >= 1:
                dp2 = max(dp2, dp1 + a[1] * b[i])
            # Update dp1 using the current value of b
            dp1 = max(dp1, a[0] * b[i])
        
        # The answer is the maximum score we can achieve using exactly 4 elements
        return dp4