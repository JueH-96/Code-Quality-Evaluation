class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Check if s can be transformed into t by any number of operations
        if sorted(s) != sorted(t):
            return 0
        
        # Find the number of positions where s can be rotated to match t
        def find_rotation_matches(s, t):
            matches = 0
            for i in range(n):
                if s[i:] + s[:i] == t:
                    matches += 1
            return matches
        
        # Calculate the number of positions where s can be rotated to match t
        matches = find_rotation_matches(s, t)
        
        # If k is 1, we can only rotate once, so return the number of matches
        if k == 1:
            return matches % MOD
        
        # Calculate the number of ways using the formula for k operations
        # We can rotate the string in a cycle of n, so we need to find the number of ways
        # to reach the target position in exactly k steps.
        # This is equivalent to finding the number of solutions to the equation:
        # (current_position + k * step_size) % n = target_position
        # where step_size is the number of positions we can rotate in one operation.
        
        # Calculate the number of ways to reach the target position in k steps
        # using the formula for combinations with repetition
        # C(k-1, matches-1) * matches
        def mod_pow(base, exp, mod):
            result = 1
            while exp > 0:
                if exp % 2 == 1:
                    result = (result * base) % mod
                base = (base * base) % mod
                exp //= 2
            return result
        
        def mod_inv(x, mod):
            return mod_pow(x, mod - 2, mod)
        
        def combinations(n, k, mod):
            if k > n:
                return 0
            num = 1
            for i in range(k):
                num = (num * (n - i)) % mod
            denom = 1
            for i in range(1, k + 1):
                denom = (denom * i) % mod
            return (num * mod_inv(denom, mod)) % mod
        
        # Calculate the number of ways to reach the target position in k steps
        result = combinations(k - 1, matches - 1, MOD) * matches % MOD
        return result