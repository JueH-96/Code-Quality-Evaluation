class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        # Subtract 1 from a decimal string (num > "0")
        def strDec(s: str) -> str:
            arr = list(s)
            i = len(arr) - 1
            while i >= 0:
                if arr[i] == '0':
                    arr[i] = '9'
                    i -= 1
                else:
                    arr[i] = str(int(arr[i]) - 1)
                    break
            # Remove leading zeros if any
            while len(arr) > 1 and arr[0] == '0':
                arr.pop(0)
            return ''.join(arr) if arr else '0'
        
        # Count how many integers in [0..num] have digit sum in [min_sum..max_sum].
        def countUpTo(num: str, minS: int, maxS: int) -> int:
            if num == '0':
                # Only number is 0; check if sum of digits (0) in range
                return 1 if 0 >= minS and 0 <= maxS else 0
            
            digits = [int(d) for d in num]
            n = len(digits)
            # dp[pos][sumSoFar][tight] = number of ways
            dp = [[[-1]*(2) for _ in range(maxS+1)] for _ in range(n+1)]
            
            def dfs(pos: int, sumSoFar: int, tight: int) -> int:
                if pos == n:
                    return 1 if minS <= sumSoFar <= maxS else 0
                if dp[pos][sumSoFar][tight] != -1:
                    return dp[pos][sumSoFar][tight]
                
                limit = digits[pos] if tight else 9
                res = 0
                
                for d in range(limit+1):
                    newSum = sumSoFar + d
                    if newSum > maxS:
                        # No point picking larger digits; sumSoFar would only increase
                        break
                    res += dfs(pos+1, newSum, tight and (d == limit))
                
                dp[pos][sumSoFar][tight] = res % MOD
                return dp[pos][sumSoFar][tight]
            
            return dfs(0, 0, 1)
        
        # Compute final result using inclusion-exclusion
        # Count of [num1..num2] = countUpTo(num2) - countUpTo(num1 - 1).
        num1_dec = strDec(num1)  # num1 - 1
        ans = countUpTo(num2, min_sum, max_sum) - countUpTo(num1_dec, min_sum, max_sum)
        ans %= MOD
        return ans