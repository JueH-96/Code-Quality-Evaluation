class Solution:
    MOD = 10**9 + 7

    def modexp(self, base: int, exp: int) -> int:
        res = 1
        base %= self.MOD
        while exp:
            if exp & 1:
                res = (res * base) % self.MOD
            base = (base * base) % self.MOD
            exp //=  2
        return res

    def stringCount(self, n: int) -> int:
        # A string s is "good" if we can rearrange its characters so that 
        # the substring "leet" appears. This is possible if and only if 
        # s contains at least one 'l', at least two 'e's, and at least one 't'.
        # In other words, if we denote:
        #    A: "s has no letter 'l'"
        #    B: "s has no letter 't'"
        #    C: "s has at most one 'e'"
        # then s is good if it avoids the union A U B U C.
        #
        # Total strings = 26^n.
        # Use inclusion–exclusion:
        #   good = 26^n 
        #          - [N(no 'l') + N(no 't') + N(e count ≤ 1)]
        #          + [N(no 'l' and no 't') + N(no 'l' and e count ≤ 1) + N(no 't' and e count ≤ 1)]
        #          - N(no 'l', no 't', and e count ≤ 1).
        #
        # We can compute:
        #   N(no 'l') = 25^n.
        #   N(no 't') = 25^n.
        #   N(e count ≤ 1) = (no 'e') + (exactly one 'e')
        #                   = 25^n + n * 25^(n-1).
        #
        #   N(no 'l' and no 't') = 24^n.
        #   N(no 'l' and e count ≤ 1):
        #         Here, letters other than 'l' are allowed (25 choices).
        #         Excluding 'e' gives 24 choices.
        #         So:
        #         = (no 'e') + (exactly one 'e')
        #         = 24^n + n * 24^(n-1).
        #   Similarly, N(no 't' and e count ≤ 1) = 24^n + n * 24^(n-1).
        #
        #   N(no 'l', no 't', and e count ≤ 1):
        #         Now allowed letters: 26 minus {'l','t'}: 24 letters.
        #         Excluding 'e' gives 23 letters.
        #         So:
        #         = (no 'e'): 23^n + (exactly one 'e'): n * 23^(n-1).
        #
        # Then using inclusion–exclusion, the answer is:
        #
        # good = 26^n 
        #        - [25^n + 25^n + (25^n + n * 25^(n-1))]
        #        + [24^n + (24^n + n * 24^(n-1)) + (24^n + n * 24^(n-1))]
        #        - [23^n + n * 23^(n-1)]
        #
        # Simplify a bit:
        #   Sum in the first bracket = 3*25^n + n * 25^(n-1)
        #   Sum in the second bracket = 3*24^n + 2*n * 24^(n-1)
        #
        # Thus, our final expression is:
        #
        #   good = 26^n - (3*25^n + n * 25^(n-1)) + (3*24^n + 2*n*24^(n-1)) - (23^n + n*23^(n-1))
        #
        # Note: For n < 4 the only possible candidate substrings of length 4 do not exist,
        # so the answer is 0.
        
        if n < 4:
            return 0
        
        mod = self.MOD

        pow26 = self.modexp(26, n)
        pow25 = self.modexp(25, n)
        pow25nm1 = (n * self.modexp(25, n - 1)) % mod
        pow24 = self.modexp(24, n)
        pow24nm1 = (n * self.modexp(24, n - 1)) % mod
        pow23 = self.modexp(23, n)
        pow23nm1 = (n * self.modexp(23, n - 1)) % mod

        term1 = pow26
        term2 = (3 * pow25 + pow25nm1) % mod             # N(no 'l')+N(no 't')+N(e <= 1)
        term3 = (3 * pow24 + 2 * pow24nm1) % mod           # intersections of two events
        term4 = (pow23 + pow23nm1) % mod                   # triple intersection

        ans = (term1 - term2 + term3 - term4) % mod
        return ans

# For testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.stringCount(4))    # Expected output 12
    # Example 2
    print(sol.stringCount(10))   # Expected output 83943898