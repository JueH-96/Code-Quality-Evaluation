MOD = 10**9 + 7

class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        if n != len(t):
            return 0

        # The only allowed operation takes a suffix of s of length l (1 ≤ l < n)
        # and puts it at the front. That is exactly a cyclic right‐rotation by l.
        # Thus, after k operations with values l1,l2,...,lk (each in {1,…,n–1})
        # the net result is a right rotation by r = (l1 + l2 + ... + lk) mod n.
        #
        # So, transforming s into t is only possible if t is a rotation of s.
        # In fact, if s rotated right by r equals t then there is a net rotation value r modulo n.
        # (Note: If s is periodic some rotations may produce the same string.)
        #
        # In our counting, the number of sequences of k operations that yield a net rotation r is 
        # the coefficient of x^r in (x + x^2 + ... + x^(n-1))^k modulo (x^n - 1).
        #
        # A key observation is that if we evaluate the polynomial
        #    P(x) = x + x^2 + ... + x^(n-1)
        # at any n-th root of unity ω (with ω != 1) we have:
        #    1 + ω + ω^2 + ... + ω^(n-1) = 0   -->   P(ω) = -1.
        # And for ω = 1 we have P(1) = n - 1.
        #
        # Thus by applying the discrete Fourier inversion, one may show that
        #  • When the net rotation r ≡ 0 (mod n), the number of sequences is
        #         f(0, k) = ((n-1)^k + (-1)^k * (n-1)) / n.
        #  • Otherwise, for r ≠ 0 (mod n), the number of sequences is
        #         f(r, k) = ((n-1)^k - (-1)^k) / n.
        #
        # To get the final answer we first need to determine all valid r for which
        # applying a rotation by r to s gives t. Note that a right rotation by r is the same as
        # a left rotation by (n - r). So if we find all left rotation offsets j (0 ≤ j < n)
        # such that:
        #    s[j:] + s[:j] == t
        # then the corresponding right rotation value is r = (n - j) mod n.
        #
        # We use the KMP algorithm to find all occurrences of t in s+s (considering only starting 
        # indices less than n).
    
        text = s + s
        pattern = t

        def build_pi(pat):
            m = len(pat)
            pi = [0] * m
            j = 0
            for i in range(1, m):
                while j > 0 and pat[i] != pat[j]:
                    j = pi[j-1]
                if pat[i] == pat[j]:
                    j += 1
                    pi[i] = j
            return pi

        pi = build_pi(pattern)
        occurrences = []
        j = 0
        m = len(pattern)
        for i in range(len(text)):
            while j > 0 and text[i] != pattern[j]:
                j = pi[j-1]
            if text[i] == pattern[j]:
                j += 1
            if j == m:
                start = i - m + 1
                if start < n:  # only consider rotations (left offsets) within [0, n-1]
                    occurrences.append(start)
                j = pi[j-1]

        # If t is not a rotation of s then there is no way to reach t.
        if not occurrences:
            return 0

        # For every occurrence (a left rotation offset j giving t),
        # the corresponding right rotation is r = (n - j) % n.
        # Count the number of times we obtain r = 0 and r ≠ 0.
        count0 = 0
        countNon0 = 0
        for j in occurrences:
            r = (n - j) % n
            if r == 0:
                count0 += 1
            else:
                countNon0 += 1

        # Now, let A = (n-1)^k mod MOD and let sign = (-1)^k mod MOD.
        A = pow(n-1, k, MOD)
        sign = 1 if (k % 2 == 0) else MOD - 1

        inv_n = pow(n, MOD-2, MOD)  # Modular inverse of n since MOD is prime.

        # Compute the number of sequences that yield net rotation 0.
        f0 = (A + sign * (n - 1)) % MOD
        f0 = (f0 * inv_n) % MOD
        # For a nonzero rotation r:
        f_non0 = (A - sign) % MOD
        f_non0 = (f_non0 * inv_n) % MOD

        # Sum over all valid r.
        total = (count0 * f0 + countNon0 * f_non0) % MOD
        return total


# ---------------------------
# For testing purposes only:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.numberOfWays("abcd", "cdab", 2))  # Expected output: 2
    # Example 2:
    print(sol.numberOfWays("ababab", "ababab", 1))  # Expected output: 2
    # Additional tests:
    print(sol.numberOfWays("ab", "ba", 1))  # Expected output: 1
    print(sol.numberOfWays("ab", "ab", 1))  # Expected output: 0
    print(sol.numberOfWays("abcd", "abcd", 2))  # Expected output: 3