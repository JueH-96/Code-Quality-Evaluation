from functools import lru_cache

class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        # Digit DP to count numbers in [0, bound] with digit sum in [min_sum, max_sum]
        def count_upto(bound: str) -> int:
            n = len(bound)
            
            @lru_cache(None)
            def dfs(pos: int, tight: bool, s: int) -> int:
                # pos: current index in bound, s: sum so far
                if pos == n:
                    # check sum constraint
                    return 1 if min_sum <= s <= max_sum else 0
                res = 0
                up = int(bound[pos]) if tight else 9
                for d in range(0, up + 1):
                    new_tight = tight and (d == up)
                    new_s = s + d
                    # prune if sum already exceeds max_sum
                    if new_s > max_sum:
                        break
                    res = (res + dfs(pos + 1, new_tight, new_s)) % MOD
                return res
            
            return dfs(0, True, 0)
        
        # helper to compute num1 - 1 as a string of same length (with leading zeros)
        def subtract_one(s: str) -> str:
            arr = list(s)
            i = len(arr) - 1
            while i >= 0:
                if arr[i] == '0':
                    arr[i] = '9'
                    i -= 1
                else:
                    arr[i] = str(int(arr[i]) - 1)
                    break
            # if the highest became '0' and it was the only digit, keep it
            # we keep leading zeros to maintain length
            return ''.join(arr)
        
        # compute f(num2) - f(num1 - 1)
        total2 = count_upto(num2)
        num1m1 = subtract_one(num1)
        total1m1 = count_upto(num1m1)
        ans = (total2 - total1m1) % MOD
        return ans