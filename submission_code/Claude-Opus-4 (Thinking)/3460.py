class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        # Create a dictionary for quick lookup of requirements
        req_dict = {end: cnt for end, cnt in requirements}
        
        # Maximum inversions we need to consider
        max_inv = max(cnt for _, cnt in requirements)
        
        # dp[i][j] = number of permutations of positions 0 to i with j inversions
        dp = [[0] * (max_inv + 1) for _ in range(n)]
        
        # Base case: position 0
        if 0 in req_dict:
            if req_dict[0] == 0:
                dp[0][0] = 1
        else:
            dp[0][0] = 1
        
        for i in range(1, n):
            if i in req_dict:
                # There's a requirement for positions 0 to i
                required = req_dict[i]
                for k in range(min(i + 1, required + 1)):
                    if required - k >= 0:
                        dp[i][required] = (dp[i][required] + dp[i-1][required - k]) % MOD
            else:
                # No requirement for positions 0 to i
                for j in range(max_inv + 1):
                    for k in range(min(i + 1, j + 1)):
                        if j - k >= 0:
                            dp[i][j] = (dp[i][j] + dp[i-1][j - k]) % MOD
        
        return dp[n-1][req_dict[n-1]]