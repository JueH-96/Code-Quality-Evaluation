class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        dp = [[0] * n for _ in range(20)]
        for i in range(n):
            dp[0][i] = receiver[i]
        
        for j in range(1, 20):
            for i in range(n):
                if dp[j-1][i] != -1:
                    dp[j][i] = dp[j-1][dp[j-1][i]]
        
        def get_value(x, k):
            res = x
            for i in range(19, -1, -1):
                if k & (1 << i):
                    res += dp[i][x]
                    x = dp[i][x]
            return res
        
        max_value = 0
        for i in range(n):
            max_value = max(max_value, get_value(i, k))
        
        return max_value