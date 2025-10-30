class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        def combinations(n, k):
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1
            if k > n // 2:
                k = n - k
            
            res = 1
            for i in range(k):
                res = (res * (n - i)) // (i + 1)
            return res
        
        if zero == 0 and one == 1:
            return 1
        if zero == 1 and one == 0:
            return 1
        
        if zero == 0 or one == 0:
            return 0

        dp = {}

        def solve(z, o, last):
            if z == 0 and o == 0:
                return 1
            
            if (z, o, last) in dp:
                return dp[(z, o, last)]
            
            ans = 0
            
            if z > 0 and last != 0:
                valid = True
                if z > limit:
                    valid = False
                if valid:
                    ans = (ans + solve(z - 1, o, 0)) % MOD
            
            if o > 0 and last != 1:
                valid = True
                if o > limit:
                    valid = False
                if valid:
                    ans = (ans + solve(z, o - 1, 1)) % MOD
            
            dp[(z, o, last)] = ans
            return ans

        ans = 0
        
        if zero <= limit:
            ans = (ans + solve(zero - 1, one, 0)) % MOD
        
        if one <= limit:
            ans = (ans + solve(zero, one - 1, 1)) % MOD
        
        
        
        def solve2(z, o):
            if z == 0 and o == 0:
                return 1
            
            if z < 0 or o < 0:
                return 0
            
            if z + o == 1:
                return 1
            
            ans = 0
            
            for i in range(1, min(z + 1, limit + 1)):
                ans = (ans + combinations(z + o - i - 1, o - 1)) % MOD
            
            for i in range(1, min(o + 1, limit + 1)):
                ans = (ans + combinations(z + o - i - 1, z - 1)) % MOD
            
            return ans
        
        
        def solve3(z, o, l):
            
            def count_stable_arrays(zero, one, limit):
                if zero == 0 and one == 1:
                    return 1
                if zero == 1 and one == 0:
                    return 1
                if zero == 0 or one == 0:
                    return 0

                n = zero + one
                count = 0
                import itertools
                for arr in itertools.permutations([0] * zero + [1] * one):
                    arr = list(arr)
                    is_stable = True
                    for i in range(n - limit):
                        sub_array = arr[i:i + limit + 1]
                        if 0 not in sub_array or 1 not in sub_array:
                            is_stable = False
                            break
                    if is_stable:
                        count += 1
                return count % (10**9 + 7)
            
            return count_stable_arrays(z, o, l)
        
        return solve3(zero, one, limit)