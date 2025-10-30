class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        # Precompute the possible values for each position
        # We need to choose 4 indices i0 < i1 < i2 < i3
        # The score is a[0]*b[i0] + a[1]*b[i1] + a[2]*b[i2] + a[3]*b[i3]
        # To maximize the score, we need to find the best combination of b[i0], b[i1], b[i2], b[i3]
        # Since the indices must be in increasing order, we can use dynamic programming
        
        # Initialize DP arrays
        # dp0[i] will store the maximum value of a[0] * b[j] for j <= i
        dp0 = [float('-inf')] * n
        dp0[0] = a[0] * b[0]
        for i in range(1, n):
            dp0[i] = max(dp0[i-1], a[0] * b[i])
        
        # dp1[i] will store the maximum value of a[0] * b[j] + a[1] * b[k] for j < k <= i
        dp1 = [float('-inf')] * n
        for i in range(1, n):
            dp1[i] = max(dp1[i-1], dp0[i-1] + a[1] * b[i])
        
        # dp2[i] will store the maximum value of a[0] * b[j] + a[1] * b[k] + a[2] * b[l] for j < k < l <= i
        dp2 = [float('-inf')] * n
        for i in range(2, n):
            dp2[i] = max(dp2[i-1], dp1[i-1] + a[2] * b[i])
        
        # dp3[i] will store the maximum value of a[0] * b[j] + a[1] * b[k] + a[2] * b[l] + a[3] * b[m] for j < k < l < m <= i
        dp3 = [float('-inf')] * n
        for i in range(3, n):
            dp3[i] = max(dp3[i-1], dp2[i-1] + a[3] * b[i])
        
        # The answer is the maximum value in dp3
        return max(dp3)