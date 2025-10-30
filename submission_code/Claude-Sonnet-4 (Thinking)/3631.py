class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Precompute steps to reduce each popcount to 1
        max_popcount = n
        steps = [0] * (max_popcount + 1)
        steps[1] = 1  # Numbers with popcount 1 (like 2,4,8...) take 1 step to reach 1
        
        for c in range(2, max_popcount + 1):
            popcount_of_c = bin(c).count('1')
            steps[c] = 1 + steps[popcount_of_c]
        
        # Digit DP
        memo = {}
        
        def dp(pos, popcount, is_limit, is_num):
            if pos == n:
                if not is_num:
                    return 0
                if is_limit:
                    return 0  # This is exactly n, exclude it
                # Check if this popcount is k-reducible
                return 1 if popcount > 0 and steps[popcount] <= k else 0
            
            if (pos, popcount, is_limit, is_num) in memo:
                return memo[(pos, popcount, is_limit, is_num)]
            
            result = 0
            
            # Option 1: Don't place any digit (leading zeros)
            if not is_num:
                result += dp(pos + 1, popcount, False, False)
            
            # Option 2: Place a digit
            start = 1 if not is_num else 0  # First digit can't be 0
            limit = int(s[pos]) if is_limit else 1
            
            for digit in range(start, limit + 1):
                new_popcount = popcount + digit
                new_is_limit = is_limit and (digit == limit)
                new_is_num = True
                
                result += dp(pos + 1, new_popcount, new_is_limit, new_is_num)
                result %= MOD
            
            memo[(pos, popcount, is_limit, is_num)] = result
            return result
        
        return dp(0, 0, True, False)