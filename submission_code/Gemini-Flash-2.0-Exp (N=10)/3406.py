class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        if one == 0:
            return 0
        if zero == 0:
            return 0 if one > 1 else 1
        
        if one == 1:
            return 2 if zero <= limit else 0
        
        dp = {}
        
        def solve(zeros_left, ones_left, last_was_one):
            if (zeros_left, ones_left, last_was_one) in dp:
                return dp[(zeros_left, ones_left, last_was_one)]
            
            if zeros_left == 0 and ones_left == 0:
                return 1
            
            ans = 0
            
            if zeros_left > 0:
                ans = (ans + solve(zeros_left - 1, ones_left, 0)) % MOD
            
            if ones_left > 0 and (last_was_one == 0 or limit == 1):
                ans = (ans + solve(zeros_left, ones_left - 1, 1)) % MOD
            
            dp[(zeros_left, ones_left, last_was_one)] = ans
            return ans
        
        
        total_ways = 0
        
        
        for i in range(1, one + 1):
            
            
            total_ways = (total_ways + solve(zero, one - i, 1)) % MOD
            
            
            
        
        
        return total_ways