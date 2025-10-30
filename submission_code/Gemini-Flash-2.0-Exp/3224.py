class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7

        def factorial(num):
            res = 1
            for i in range(1, num + 1):
                res = (res * i) % MOD
            return res

        def power(base, exp):
            res = 1
            base %= MOD
            while exp > 0:
                if exp % 2 == 1:
                    res = (res * base) % MOD
                base = (base * base) % MOD
                exp //= 2
            return res

        def inverse(num):
            return power(num, MOD - 2)

        def combinations(n, k):
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1
            if k > n // 2:
                k = n - k
            
            numerator = factorial(n)
            denominator = (factorial(k) * factorial(n - k)) % MOD
            return (numerator * inverse(denominator)) % MOD

        healthy = n - len(sick)
        
        if healthy == 0:
            return 1

        result = 1
        
        # Calculate combinations for the gaps between sick children
        for i in range(len(sick) - 1):
            gap = sick[i+1] - sick[i] - 1
            if gap > 0:
                result = (result * combinations(healthy, gap)) % MOD
                result = (result * power(2, gap - 1 if i < len(sick) - 2 else gap -1 )) % MOD
                healthy -= gap
        
        # Calculate the number of ways to arrange the remaining healthy children
        result = (result * factorial(healthy)) % MOD
        
        # Handle the leading and trailing healthy children
        leading = sick[0]
        trailing = n - 1 - sick[-1]
        
        return result