class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        # Find positions where s1 and s2 differ
        diff_pos = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                diff_pos.append(i)
        
        # If odd number of differences, impossible to make equal
        if len(diff_pos) % 2:
            return -1
            
        # If no differences, already equal
        if not diff_pos:
            return 0
            
        n = len(diff_pos)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        if n >= 1:
            dp[1] = float('inf')
        
        if n >= 2:
            dp[2] = min(x, diff_pos[1] - diff_pos[0])
            
        for i in range(3, n + 1):
            # Either use operation 1 (cost x) to fix first difference with any other
            dp[i] = dp[i-2] + x
            
            # Or use operation 2 (cost 1) to fix adjacent differences
            if diff_pos[i-1] - diff_pos[i-2] == 1:
                dp[i] = min(dp[i], dp[i-2] + 1)
                
        return dp[n] if dp[n] != float('inf') else -1