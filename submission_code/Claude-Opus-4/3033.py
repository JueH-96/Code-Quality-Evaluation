class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        # Find all positions where s1 and s2 differ
        diff_positions = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                diff_positions.append(i)
        
        # If odd number of differences, impossible to make equal
        if len(diff_positions) % 2 == 1:
            return -1
        
        # If no differences, already equal
        if len(diff_positions) == 0:
            return 0
        
        n = len(diff_positions)
        
        # dp[i][j] = minimum cost to fix first i differences with j unpaired
        # j can be 0 or 1
        INF = float('inf')
        dp = [[INF] * 2 for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            # If we pair current difference with a previous unpaired one using operation 1
            if i >= 2 and diff_positions[i-1] - diff_positions[i-2] == 1:
                # Adjacent positions - can use operation 2 (cost 1)
                dp[i][0] = min(dp[i][0], dp[i-2][0] + 1)
            
            # Pair with previous unpaired using operation 1 (cost x)
            dp[i][0] = min(dp[i][0], dp[i-1][1] + x)
            
            # Leave current unpaired
            dp[i][1] = dp[i-1][0]
        
        return dp[n][0]