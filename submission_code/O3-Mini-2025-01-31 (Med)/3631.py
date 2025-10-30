MOD = 10**9 + 7

class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        # If n == 1 then there is no positive integer less than n.
        if s == "1":
            return 0

        L = len(s)
        
        # Define a helper function f(x) which returns the minimum number of popcount operations
        # required to reduce x to 1.
        # f(1) = 0; for x > 1, f(x) = 1 + f(popcount(x)).
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def f(x: int) -> int:
            if x == 1:
                return 0
            return 1 + f(bin(x).count("1"))
        
        # Precompute factorials and inverse factorials to quickly compute combinations.
        # In our counting below, the maximum we need is up to L (number of bits) and L <= 800.
        N = L + 10  # a little extra
        fact = [1] * (N + 1)
        invfact = [1] * (N + 1)
        for i in range(2, N + 1):
            fact[i] = fact[i - 1] * i % MOD
        invfact[N] = pow(fact[N], MOD - 2, MOD)
        for i in range(N, 0, -1):
            invfact[i - 1] = invfact[i] * i % MOD

        def nCr(n: int, r: int) -> int:
            if r < 0 or r > n:
                return 0
            return fact[n] * invfact[r] % MOD * invfact[n - r] % MOD
        
        # count_exact(r, s) returns the number of positive integers (with possible leading zeros
        # in their L-bit representation) that are strictly less than n (with binary representation s)
        # and have exactly r ones.
        # The idea is to iterate over each bit of s. When we see a '1', we have the option of putting
        # a '0' in that position and then arbitrarily choosing (r - (number of ones chosen so far))
        # ones among the remaining bits. This counts all numbers that are definitely less than s.
        def count_exact(r: int, s: str) -> int:
            ways = 0
            ones = 0
            L = len(s)
            for i in range(L):
                if s[i] == '1':
                    # We choose 0 here even though s has a 1.
                    ways = (ways + nCr(L - i - 1, r - ones)) % MOD
                    ones += 1
                    if ones > r:
                        break
            return ways

        # Now, an integer x is "k‑reducible" if applying the popcount operation at most k times 
        # reduces it to 1.
        # For any integer x > 1, let r = popcount(x). Then one operation takes x -> r.
        # Thus, for x > 1, we need: 1 + f(r) <= k  i.e.  f(r) <= k-1.
        # (The special case x == 1 is already k‑reducible since no operation is needed.)
        #
        # We will “group” numbers x (1 <= x < n) by their number of ones, r. Notice that if x is a power of 2,
        # it has exactly one set bit; however the number 1, which is a power of 2, has f(1)=0 while any other
        # power of 2 has f(x)=1. In either case, if k >= 1 then x qualifies.
        #
        # So overall we sum over r = 1 to L:
        #   • When r == 1, every x with exactly one '1' qualifies (since f(x) is either 0 or 1 and k>=1).
        #   • For r > 1, x qualifies if f(r) <= k - 1.
        result = 0
        for r in range(1, L + 1):
            if r == 1:
                result = (result + count_exact(1, s)) % MOD
            else:
                if f(r) <= k - 1:
                    result = (result + count_exact(r, s)) % MOD
        
        return result % MOD


# ---------------------------- Testing Code ----------------------------

if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    # s = "111" represents n = 7. The 1‑reducible numbers less than 7 are 1, 2, and 4.
    print(sol.countKReducibleNumbers("111", 1))  # Expected output: 3

    # Example 2:
    # s = "1000" represents n = 8. The 2‑reducible numbers less than 8 are 1, 2, 3, 4, 5, and 6.
    print(sol.countKReducibleNumbers("1000", 2))  # Expected output: 6

    # Example 3:
    # s = "1" represents n = 1 so there are no positive integers less than n.
    print(sol.countKReducibleNumbers("1", 3))     # Expected output: 0