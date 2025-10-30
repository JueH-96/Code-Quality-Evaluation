class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        mod = 10**9 + 7

        # Compute the minimal rotation needed to get from s to t
        def compute_shift(s, t):
            combined = s + s
            idx = combined.find(t, 1, n)
            if idx == -1:
                return -1
            else:
                return idx
        shift = compute_shift(s, t)
        if shift == -1:
            return 0

        from math import gcd

        # Calculate the number of positions n where rotating s gives t
        count = 0
        total_gcd = gcd(n, shift)
        minimal_rotation = n // total_gcd

        # Use modular exponentiation to compute the number of ways
        # Since the rotations form a cyclic group, the number of ways is:
        # Number of times shift divides into k
        if k % minimal_rotation != 0:
            return 0
        else:
            m = k // minimal_rotation
            result = pow(total_gcd, m, mod)
            return result % mod