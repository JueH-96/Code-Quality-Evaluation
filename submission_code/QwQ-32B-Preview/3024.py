class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Function to compute (a^b) % MOD
        def powmod(a, b):
            result = 1
            a = a % MOD
            while b > 0:
                if b % 2:
                    result = (result * a) % MOD
                a = (a * a) % MOD
                b //= 2
            return result
        
        # Find all r where rotate(s, r) == t
        def find_valid_r(s, t, n):
            # Concatenate s with itself
            s2 = s + s
            # Find all starting positions where t is a substring
            valid_r = []
            start = 0
            while True:
                start = s2.find(t, start)
                if start == -1:
                    break
                valid_r.append(start)
                start += 1
            # Convert positions to rotation amounts
            return [r for r in valid_r if r < n]
        
        valid_r = find_valid_r(s, t, n)
        
        # Compute the number of sequences for each r
        total = 0
        m = n - 1
        mk = powmod(m, k)
        neg1k = 1 if k % 2 == 0 else -1
        for r in valid_r:
            if r == 0:
                # For r == 0
                count = (mk + neg1k * m) // n
            else:
                # For r != 0
                count = (mk - neg1k) // n
            total = (total + count) % MOD
        
        return total