class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        """
        We want the number of ways to transform s into t in exactly k operations,
        where each operation consists of removing a nonempty proper suffix of s
        and appending it to the front. Equivalently, each operation is a "rotation
        to the right by l" (1 <= l < n). We count how many sequences of exactly k
        such rotations convert s into t, modulo 1e9+7.
        
        Key observations:
        
        1) Each rotation by l (1 <= l < n) is effectively moving from "offset i" to "offset i + l (mod n)" 
           in a space of n possible offsets, where offset 0 corresponds to the original string s,
           offset 1 to s rotated right by 1, etc.
           
        2) From any offset i, we can jump to any other offset j != i, so there are (n-1) possible moves
           from each state. This corresponds to the (n x n) adjacency matrix M with:
                M[i, j] = 1 if j != i, else 0
           i.e. M = J_n - I_n, where J_n is the all-ones matrix and I_n is the identity.

        3) We need (M^k)[0, r], i.e. the number of ways to go from offset 0 to offset r in exactly k steps.
           Because M = J_n - I_n, one can show (by diagonalizing or a direct commuting argument) that
           for all i != j:
               (M^k)[i, j] = S_off(n, k) = [ (n-1)^k - (-1)^k ] / n
           and for the diagonal entries i == j:
               (M^k)[i, i] = S_diag(n, k) = [ (n-1)^k + (n-1)*(-1)^k ] / n
           (all arithmetic done mod 10^9+7).
           
        4) To know which offset r corresponds to s rotated right by r == t, we look for all
           r in [0..n-1] such that rotating s right by r gives t. Equivalently, t appears at
           index (n-r) in the doubled string s+s (because a right-rotation by r places what
           used to be s[n-r..n-1] at the front). We can find all such occurrences using KMP
           (or Z-alg) in O(n) time.

        5) Let R be the set of all offsets r that produce t. Then the answer is 
              sum_{r in R} (M^k)[0, r].
           Because each r in R is either 0 or not 0, the sum is
              (#offsets in R that are not 0) * S_off(n, k) + (1 if 0 in R else 0) * S_diag(n, k).
           
        6) We compute (n-1)^k mod 10^9+7 by fast exponentiation (O(log k)). Also compute (-1)^k mod.
           Then multiply/divide (using modular inverse of n) accordingly.

        Steps:
        - If s and t are not the same length or are trivial, handle edge cases accordingly.
        - Build s2 = s + s and use KMP (or similar) to find all indices i in [0..n-1] with s2[i..i+n-1] = t.
          For each such i, the corresponding rotation offset is (n - i) mod n.
        - Compute (M^k)[0,0] = S_diag(n, k) and (M^k)[0, any_other] = S_off(n, k) using the closed-form.
        - Sum over all valid offsets to get the final answer.
        """

        import sys
        sys.setrecursionlimit(10**7)
        MOD = 10**9 + 7

        n = len(s)
        # Quick check (though problem states s.length == t.length)
        if n != len(t):
            return 0

        # KMP to find all occurrences of t in s+s
        # We only care about start indices i in [0..n-1].
        s2 = s + s

        # Build the KMP "lps" array for pattern t
        lps = [0]*n
        def build_lps(pattern):
            j = 0
            for i in range(1, len(pattern)):
                while j > 0 and pattern[i] != pattern[j]:
                    j = lps[j-1]
                if pattern[i] == pattern[j]:
                    j += 1
                    lps[i] = j

        build_lps(t)

        # KMP matching on s2 with pattern t
        R = set()
        j = 0  # index into t
        for i in range(2*n):
            while j > 0 and s2[i] != t[j]:
                j = lps[j-1]
            if s2[i] == t[j]:
                j += 1
            if j == n:
                start_pos = i - n + 1
                if start_pos < n:  # only take occurrences within [0..n-1]
                    # offset r = (n - start_pos) mod n
                    offset = (n - start_pos) % n
                    R.add(offset)
                j = lps[j-1]
            if i+1 == 2*n:
                break

        # If no valid offsets, answer is 0
        if not R:
            return 0

        # Precompute needed powers and inverses
        def modexp(base, exp, m):
            """Fast exponentiation base^exp mod m."""
            result = 1
            cur = base % m
            e = exp
            while e > 0:
                if e & 1:
                    result = (result * cur) % m
                cur = (cur * cur) % m
                e >>= 1
            return result

        # Modular inverse of n w.r.t. MOD (since MOD is prime)
        def inv(x):
            return pow(x, MOD-2, MOD)

        c = modexp(n-1, k, MOD)               # (n-1)^k mod
        d = 1 if (k % 2 == 0) else (MOD - 1)  # (-1)^k mod  => 1 if even, MOD-1 if odd
        inv_n = inv(n)

        # (M^k)[i,j] for i != j => S_off = [ (n-1)^k - (-1)^k ] / n
        S_off = ( (c - d) % MOD ) * inv_n % MOD

        # (M^k)[i,i] => S_diag = [ (n-1)^k + (n-1)*(-1)^k ] / n
        # = [ c + (n-1)*d ] / n
        S_diag = ( (c + ( (n-1) * d ) % MOD ) % MOD ) * inv_n % MOD

        m = len(R)
        r0 = 1 if (0 in R) else 0

        # Answer = r0 * S_diag + (m - r0) * S_off (mod)
        ans = (r0 * S_diag + (m - r0) * S_off) % MOD
        return ans