class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        mod = 10**9 + 7
        req = [-1] * n
        for e, cnt in requirements:
            req[e] = cnt
        
        dp_prev = [1]
        
        for i in range(1, n + 1):
            max_curr = i * (i - 1) // 2
            new_dp = [0] * (max_curr + 1)
            window = 0
            for j in range(0, max_curr + 1):
                if j < len(dp_prev):
                    window = (window + dp_prev[j])
                if j - i >= 0 and j - i < len(dp_prev):
                    window = (window - dp_prev[j - i])
                window %= mod
                new_dp[j] = window
            
            if i - 1 < n and req[i - 1] != -1:
                c = req[i - 1]
                tmp = [0] * (max_curr + 1)
                if 0 <= c <= max_curr:
                    tmp[c] = new_dp[c]
                new_dp = tmp
            
            dp_prev = new_dp
        
        return dp_prev[req[n - 1]] % mod