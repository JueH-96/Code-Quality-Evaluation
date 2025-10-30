class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        # Sort requirements by end index
        requirements.sort()
        
        # Initialize dp array
        dp = [[[0] * (n + 1) for _ in range(n + 1)] for _ in range(n + 1)]
        dp[0][0][0] = 1
        
        for i in range(1, n + 1):
            for j in range(i + 1):
                for k in range(i + 1):
                    # Case 1: Put i at the end
                    dp[i][j][k] = dp[i-1][j][k] * (i - j)
                    
                    # Case 2: Put i before the end
                    if j > 0 and k > 0:
                        dp[i][j][k] += dp[i-1][j-1][k-1] * j
                    
                    dp[i][j][k] %= MOD
        
        result = dp[n][n][requirements[0][1]]
        
        for i in range(1, len(requirements)):
            end, cnt = requirements[i]
            prev_end = requirements[i-1][0]
            length = end - prev_end
            inversions = cnt - requirements[i-1][1]
            
            if inversions < 0 or inversions > length * (length - 1) // 2:
                return 0
            
            new_result = 0
            for j in range(length + 1):
                new_result += dp[length][j][inversions] * result
                new_result %= MOD
            
            result = new_result
        
        return result