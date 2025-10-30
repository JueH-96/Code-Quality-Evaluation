class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        if k > n - 1:
            return 0
        if n == 1:
            return m
        
        dp = {}
        
        def solve(index, consecutive_matches, prev_val):
            if index == n:
                if consecutive_matches == k:
                    return 1
                else:
                    return 0
            
            if (index, consecutive_matches, prev_val) in dp:
                return dp[(index, consecutive_matches, prev_val)]
            
            ans = 0
            for curr_val in range(1, m + 1):
                new_consecutive_matches = consecutive_matches
                if index > 0 and curr_val == prev_val:
                    new_consecutive_matches += 1
                
                ans = (ans + solve(index + 1, new_consecutive_matches, curr_val)) % MOD
            
            dp[(index, consecutive_matches, prev_val)] = ans
            return ans
        
        return solve(0, 0, 0)