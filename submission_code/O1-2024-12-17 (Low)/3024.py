class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        """
        We want the number of ways to transform s into t in exactly k operations,
        where each operation "cuts off" a nonempty proper suffix of s and puts it
        in front. Equivalently, if we think of s as a 'rotating offset' around a
        circle of length n, each operation increases that offset by l (1 <= l < n)
        modulo n. We want the offset after k steps to align so that the rotated s
        equals t.

        Key steps:
          1) Find all offsets d in [0..n-1] for which rotating s by d yields t.
             We'll call the set of these offsets A. If no such offset exists, answer is 0.
          2) Let M = 10^9+7. The number of ways to go from offset 0 to offset j in exactly
             k steps (with each step adding a nonzero mod-n amount) is given by (M^k e_0)_j,
             where M is the n×n matrix of all 1's minus identity (so from i you can go to
             any j != i).
             Using eigen-decomposition, one can derive that:
               w_0 = ( (n-1)^k + (-1)^k*(n-1) ) / n   (for offset 0)
               w_j = ( (n-1)^k - (-1)^k ) / n         (for offset j != 0)
          3) Sum these values w_j over j in the set A, under modulo 10^9+7.

        Explanation of formula for M^k e_0:
          - Matrix of all ones J has eigenvalues {n, 0, 0, ..., 0}.
          - Thus (J - I) has eigenvalues {n-1, -1, -1, ..., -1}.
          - One can show in closed form what the (0,j) entries of (J - I)^k are.
        """

        MOD = 10**9 + 7
        n = len(s)

        # Step 1: Find all valid offsets d where rotating s by d == t
        # We look for occurrences of t in (s + s).
        # Standard KMP or other string-search can do this in O(n).
        # We'll record how many such offsets, and whether 0 is among them.

        # Quick check: if lengths differ, return 0 (not in problem constraints but a simple guard).
        if len(t) != n:
            return 0

        # KMP to find pattern t in text = s+s
        text = s + s
        pattern = t

        # Build KMP "lps" (longest proper prefix-suffix) array for pattern
        lps = [0]*n
        j = 0  # length of the previous longest prefix suffix
        i = 1
        while i < n:
            if pattern[i] == pattern[j]:
                j += 1
                lps[i] = j
                i += 1
            else:
                if j != 0:
                    j = lps[j-1]
                else:
                    lps[i] = 0
                    i += 1

        # KMP matching
        matches_count = 0
        zero_in_A = False

        i = 0  # index for text
        j = 0  # index for pattern
        while i < 2*n:
            if text[i] == pattern[j]:
                i += 1
                j += 1
                if j == n:
                    # match ends at i-1 => match start is i-n
                    start_idx = i - n
                    if start_idx < n:
                        matches_count += 1
                        if start_idx == 0:
                            zero_in_A = True
                    j = lps[j-1]
            else:
                if j != 0:
                    j = lps[j-1]
                else:
                    i += 1

        if matches_count == 0:
            return 0  # no offset yields t

        # Step 2: Compute (n-1)^k mod, and (-1)^k mod
        # (-1)^k mod => 1 if k even; -1 mod => MOD-1 if k odd
        pow_n_minus_1 = pow(n-1, k, MOD)
        if k % 2 == 0:
            neg_one_pow_k = 1
        else:
            neg_one_pow_k = MOD - 1  # effectively -1 mod M

        inv_n = pow(n, MOD-2, MOD)  # modular inverse of n

        # w_0  = [ (n-1)^k + (n-1)*(-1)^k ] / n
        # w_j != 0 = [ (n-1)^k - (-1)^k ] / n
        # Then sum up for all offsets in A.

        # Precompute these two quantities:
        w_0_num = (pow_n_minus_1 + ((n-1) * neg_one_pow_k) % MOD) % MOD
        w_0_val = (w_0_num * inv_n) % MOD

        w_j_num = (pow_n_minus_1 - neg_one_pow_k) % MOD
        w_j_val = (w_j_num * inv_n) % MOD

        # If 0 in A, sum = w_0 + (c-1)*w_j
        # Else sum = c*w_j
        c = matches_count
        if zero_in_A:
            return (w_0_val + (c - 1) * w_j_val) % MOD
        else:
            return (c * w_j_val) % MOD