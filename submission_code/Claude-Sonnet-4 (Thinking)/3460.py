class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        req_map = {}
        for end, cnt in requirements:
            req_map[end] = cnt
        
        max_inversions = 400
        dp = [[0] * (max_inversions + 1) for _ in range(n)]
        
        # Base case
        if 0 in req_map:
            if req_map[0] == 0:
                dp[0][0] = 1
            else:
                return 0
        else:
            dp[0][0] = 1
        
        for i in range(1, n):
            if i in req_map:
                required = req_map[i]
                for k in range(i + 1):
                    if required >= k:
                        dp[i][required] = (dp[i][required] + dp[i-1][required-k]) % MOD
            else:
                for j in range(max_inversions + 1):
                    for k in range(i + 1):
                        if j >= k:
                            dp[i][j] = (dp[i][j] + dp[i-1][j-k]) % MOD
        
        return dp[n-1][req_map[n-1]]