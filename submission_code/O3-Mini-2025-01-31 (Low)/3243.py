class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        # Explanation:
        # A candidate powerful integer is built by having "s" as a suffix.
        # Write candidate = prefix * 10^m + base, where base = int(s) and m = len(s).
        # In the case where no prefix is added (i.e. prefix is “empty”), candidate = base.
        # (We already know that s’s digits are valid by the problem constraints.)
        #
        # For a non-empty prefix, we want the integer “prefix” (when written in base 10)
        # to have exactly n digits (with n>=1) such that:
        #   start <= prefix * 10^m + base <= finish.
        # In addition, every digit in prefix must be at most limit.
        # (Careful: when prefix is non-empty its first digit must be nonzero.)
        #
        # We will iterate over possible prefix length n. If no prefix, we check candidate = base.
        # For each fixed number of digits n, note:
        #   p must lie between 10^(n-1) and 10^n - 1 (n-digit numbers)
        # and by the candidate range condition:
        #   p must be in [ceil((start - base)/10^m), floor((finish - base)/10^m)].
        # We need to count only those p which (as an n-digit number) use only digits from
        # {0,1,...,limit} (with the first digit from 1..limit).
        #
        # Instead of generating each candidate we count them by digit DP.
        
        base = int(s)
        m = len(s)
        ans = 0
        ten_m = 10**m

        # Case 1: no prefix (i.e. candidate equals s itself)
        if start <= base <= finish:
            ans += 1

        # For candidates with non-empty prefix:
        # Candidate = p * 10^m + base. Candidate length = (number of digits in p) + m.
        # For candidate to be at most finish, its length cannot exceed len(str(finish)).
        L = len(str(finish))
        # Let n be the number of digits in prefix. Then n + m <= L.
        max_n = L - m  
        if max_n < 1:
            return ans

        # We write a helper digit DP function which counts, for a given upper bound X,
        # the number of valid n-digit numbers built only with digits in allowed set.
        # Allowed digits are from 0 to limit, with the extra requirement that the first digit
        # is in 1..limit.
        from functools import lru_cache

        # count_exact(n, X) counts valid numbers (of exactly n digits and composed of allowed digits)
        # that are <= X. (If X has more than n digits, then every such n-digit number qualifies.)
        def count_exact(n: int, X: int) -> int:
            sX = str(X)
            if len(sX) < n:
                return 0
            if len(sX) > n:
                # Every n-digit valid number is <= X.
                return limit * ((limit+1) ** (n-1))
            
            # Otherwise, len(sX)==n; we count using our DP.
            digits = list(map(int, sX))
            @lru_cache(maxsize=None)
            def dp(pos: int, tight: bool) -> int:
                if pos == n:
                    return 1
                total = 0
                # For the first digit, disallow 0.
                low = 1 if pos == 0 else 0
                up = digits[pos] if tight else limit
                for d in range(low, up + 1):
                    if d > limit:
                        break  # because allowed digits are only up to limit.
                    ntight = tight and (d == up)
                    total += dp(pos+1, ntight)
                return total
            return dp(0, True)
        
        # count_range(n, L_bound, R_bound) will count valid n-digit numbers in the range [L_bound, R_bound]
        def count_range(n: int, L_bound: int, R_bound: int) -> int:
            if L_bound > R_bound:
                return 0
            return count_exact(n, R_bound) - count_exact(n, L_bound - 1)
        
        # Now, iterate over possible prefix lengths n = 1 to max_n.
        for n in range(1, max_n + 1):
            # All n-digit numbers are within [10^(n-1), 10^n - 1]
            p_min_possible = 10**(n-1)
            p_max_possible = 10**n - 1
            # From candidate constraints:
            #   start <= p*ten_m + base <= finish
            # so p must be in:
            #   p >= ceil((start - base) / ten_m)
            #   p <= floor((finish - base) / ten_m)
            low_p = (start - base + ten_m - 1) // ten_m  # ceil division
            high_p = (finish - base) // ten_m
            # The effective range for p is the intersection:
            effective_low = max(p_min_possible, low_p)
            effective_high = min(p_max_possible, high_p)
            if effective_low > effective_high:
                continue
            cnt = count_range(n, effective_low, effective_high)
            ans += cnt
        return ans

# For local testing
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.numberOfPowerfulInt(1, 6000, 4, "124"))  # Expected output: 5
    # Example 2:
    print(sol.numberOfPowerfulInt(15, 215, 6, "10"))   # Expected output: 2
    # Example 3:
    print(sol.numberOfPowerfulInt(1000, 2000, 4, "3000"))  # Expected output: 0