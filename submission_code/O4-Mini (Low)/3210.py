class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        # Helper: factorize k into prime powers
        def prime_factors(n):
            i = 2
            factors = {}
            while i * i <= n:
                while n % i == 0:
                    factors[i] = factors.get(i, 0) + 1
                    n //= i
                i += 1
            if n > 1:
                factors[n] = factors.get(n, 0) + 1
            return factors
        
        # Compute the minimal m such that for each p^e in k,
        # 2*vp(m) >= e  => vp(m) >= ceil(e/2)
        pf = prime_factors(k)
        m = 1
        for p, e in pf.items():
            need = (e + 1) // 2
            m *= p**need
        
        n = len(s)
        # If even the smallest t=m is too big, no substrings qualify
        if m > n // 2:
            return 0
        
        # Build prefix sums of vowels and consonants
        vowels = set('aeiou')
        pre_v = [0] * (n+1)
        pre_c = [0] * (n+1)
        for i, ch in enumerate(s, start=1):
            pre_v[i] = pre_v[i-1] + (1 if ch in vowels else 0)
            pre_c[i] = pre_c[i-1] + (1 if ch not in vowels else 0)
        
        ans = 0
        # t must be a multiple of m
        # and t <= n//2
        t = m
        while t <= n // 2:
            length = 2 * t
            # slide window of size length
            for start in range(0, n - length + 1):
                end = start + length
                vcnt = pre_v[end] - pre_v[start]
                ccnt = pre_c[end] - pre_c[start]
                # check equality (v == c == t)
                if vcnt == t and ccnt == t:
                    ans += 1
            t += m
        
        return ans