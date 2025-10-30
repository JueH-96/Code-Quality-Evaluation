class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        # If n==1 then there are no positive integers less than n.
        if s == "1":
            return 0
        L = len(s)
        
        # Precompute factorials and inverse factorials up to a safe limit.
        max_n = L + 10  # safe upper bound since L<=800
        fact = [1] * max_n
        inv_fact = [1] * max_n
        for i in range(1, max_n):
            fact[i] = (fact[i-1] * i) % MOD
        inv_fact[max_n-1] = pow(fact[max_n-1], MOD-2, MOD)
        for i in range(max_n-1, 0, -1):
            inv_fact[i-1] = (inv_fact[i] * i) % MOD
        
        def comb(n, r):
            if r < 0 or r > n:
                return 0
            return (fact[n] * inv_fact[r] % MOD) * inv_fact[n - r] % MOD
        
        # This helper counts positive integers < n (with n given in binary string n_str)
        # that have exactly r ones in their binary representation.
        # We split the count into numbers with fewer than len(n_str) digits plus numbers with exactly len(n_str) digits.
        def count_with_r(n_str, r):
            L_local = len(n_str)
            total = 0
            # Count m-digit numbers for m from 1 to L_local-1.
            for m in range(1, L_local):
                if r <= m:
                    # An m-digit binary number (without leading zeros) always starts with '1',
                    # so the remaining m-1 positions must contain (r-1) ones.
                    total = (total + comb(m-1, r-1)) % MOD
            
            # Now count L_local-digit numbers that are <= n_str.
            # (We follow the standard “digit-DP” idea using combinatorics.)
            count_L = 0
            # The most‐significant digit is fixed as '1' so we start with one 1 already.
            ones_used = 1
            for i in range(1, L_local):
                if n_str[i] == '1':
                    remaining = L_local - i - 1
                    need = r - ones_used  # still need to place these many ones in the remaining digits
                    count_L = (count_L + comb(remaining, need)) % MOD
                    ones_used += 1  # and follow the digit (i.e. treat it as a “1”)
                else:
                    # If the digit is '0' there is no “branch” option.
                    pass
            # After processing all digits, the number exactly equal to n_str is counted by this procedure.
            if ones_used == r:
                count_L = (count_L + 1) % MOD
            # We want numbers strictly less than n, so if n itself has exactly r ones, subtract one.
            if n_str.count('1') == r:
                count_L = (count_L - 1) % MOD
            return (total + count_L) % MOD
        
        # For any positive integer x, define the “reduction‐cost” as follows:
        #   • If x == 1: cost = 0.
        #   • Otherwise: cost(x) = 1 + cost(popcount(x)).
        # For any x > 1, if popcount(x) = r then cost(x) = 1 + cost(r).
        # Thus, x is called k–reducible if cost(x) <= k.
        #
        # Our plan: for any candidate x (1 <= x < n) its “property” depends only on
        # r = popcount(x). (Note that x==1 has r==1 but cost(1)==0.)
        #
        # We precompute, for small numbers v (v up to L, because the maximum possible popcount is at most L),
        # the cost defined by:
        #    cost(1) = 0, and for v > 1: cost(v) = 1 + cost( popcount(v) ).
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def cost(v: int) -> int:
            if v == 1:
                return 0
            # Use bit_count if available (Python 3.10+) otherwise fallback to bin(x).count("1").
            cnt = v.bit_count() if hasattr(v, "bit_count") else bin(v).count("1")
            return 1 + cost(cnt)
        
        # Now we sum over all possible values of r (the number of ones in x).
        # Notice: for x > 1, cost(x) = 1 + cost(r) so we require 1 + cost(r) <= k, i.e. cost(r) <= k-1.
        # And for x==1 (which also has r == 1) cost(1) == 0.
        ans = 0
        for r in range(1, L+1):
            allowed = False
            if r == 1:
                # All numbers with one '1' (i.e. powers of 2) are k–reducible if k>=1.
                # (When k==0 only 1 would qualify; given constraints k>=1 this branch works.)
                if k >= 1:
                    allowed = True
            else:
                if cost(r) <= k - 1:
                    allowed = True
            if allowed:
                cnt_r = count_with_r(s, r)
                ans = (ans + cnt_r) % MOD
        return ans % MOD

# For testing purposes:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    # n = 7 (s = "111"), k = 1.
    # The 1–reducible numbers (<7) are: 1, 2, 4  → output: 3.
    print(sol.countKReducibleNumbers("111", 1))   # Expected output: 3
    
    # Example 2:
    # n = 8 (s = "1000"), k = 2.
    # The 2–reducible numbers (<8) are: 1, 2, 3, 4, 5, 6  → output: 6.
    print(sol.countKReducibleNumbers("1000", 2))  # Expected output: 6
    
    # Example 3:
    # n = 1 (s = "1"), k = 3. No positive integer less than 1.
    print(sol.countKReducibleNumbers("1", 3))     # Expected output: 0