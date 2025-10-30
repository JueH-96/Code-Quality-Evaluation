class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[i][j] = dict mapping (min_val, max_val) to count of subsequences
        dp = [[{} for _ in range(k+1)] for _ in range(n+1)]
        
        for i in range(1, n+1):
            val = nums[i-1]
            
            for j in range(k+1):
                # Don't include nums[i-1]: copy from dp[i-1][j]
                for (min_val, max_val), count in dp[i-1][j].items():
                    if (min_val, max_val) not in dp[i][j]:
                        dp[i][j][(min_val, max_val)] = 0
                    dp[i][j][(min_val, max_val)] += count
                    dp[i][j][(min_val, max_val)] %= MOD
                
                # Include nums[i-1]
                if j == 1:
                    # Subsequence of size 1
                    if (val, val) not in dp[i][j]:
                        dp[i][j][(val, val)] = 0
                    dp[i][j][(val, val)] += 1
                    dp[i][j][(val, val)] %= MOD
                elif j > 1:
                    # Extend existing subsequences
                    for (min_val, max_val), count in dp[i-1][j-1].items():
                        new_min = min(min_val, val)
                        new_max = max(max_val, val)
                        
                        if (new_min, new_max) not in dp[i][j]:
                            dp[i][j][(new_min, new_max)] = 0
                        dp[i][j][(new_min, new_max)] += count
                        dp[i][j][(new_min, new_max)] %= MOD
        
        result = 0
        for j in range(1, k+1):
            for (min_val, max_val), count in dp[n][j].items():
                result += count * (min_val + max_val)
                result %= MOD
        
        return result