class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        h_cuts = sorted(horizontalCut, reverse=True)
        v_cuts = sorted(verticalCut, reverse=True)
        len_h = len(h_cuts)
        len_v = len(v_cuts)
        
        dp = [[float('inf')] * (len_v + 1) for _ in range(len_h + 1)]
        dp[0][0] = 0
        
        for i in range(len_h + 1):
            for j in range(len_v + 1):
                if i < len_h:
                    new_val = dp[i][j] + h_cuts[i] * (j + 1)
                    if new_val < dp[i+1][j]:
                        dp[i+1][j] = new_val
                if j < len_v:
                    new_val = dp[i][j] + v_cuts[j] * (i + 1)
                    if new_val < dp[i][j+1]:
                        dp[i][j+1] = new_val
        
        return dp[len_h][len_v]