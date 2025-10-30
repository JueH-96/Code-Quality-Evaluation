class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        # Explanation:
        # --------------
        # In one operation, we remove a suffix of length l (with 1 ≤ l < n) and put it in front.
        # This is equivalent to performing a right rotation by l.
        # Since rotations form a cyclic group modulo n, k operations correspond to adding
        # k rotation amounts (each in {1,..., n-1}) modulo n.
        #
        # Let a move be a nonzero element r (1 <= r <= n-1). Then the total net rotation, after
        # k moves, is R mod n. Note that R can be 0 (even though a single move cannot be a rotation
        # by 0, several moves can add up to 0 modulo n).
        #
        # We want the number of ways to make exactly k moves so that the net effect transforms s into t.
        # Notice that s transformed by a right rotation by R equals t if and only if t is a rotation of s.
        # More precisely, if r is the required total rotation (i.e. s rotated right by r equals t),
        # then all sequences of moves with total r (mod n) are valid.
        #
        # How many sequences of k moves develop a net rotation of r?
        # Let f(x) = sum_{l=1}^{n-1} x^l.
        # Then the number of sequences that yield a net rotation r is the coefficient of x^r in [f(x)]^k,
        # but with the twist that exponents are taken modulo n (since x^n = 1).
        #
        # If we evaluate f(x) at an nth root of unity ω, then for ω ≠ 1:
        #     f(ω) = ω + ω^2 + ... + ω^(n-1) = -1.
        # And f(1) = n-1.
        #
        # Thus, by discrete Fourier (or character) inversion, one obtains:
        #    T(0) = (1/n)*[ (n-1)^k + (-1)^k*(n-1) ]
        # and for any r ≠ 0:
        #    T(r) = (1/n)*[ (n-1)^k - (-1)^k ].
        #
        # Next, we must determine for which r (0 ≤ r < n) we have
        #   s rotated right by r equals t.
        #
        # Observe that rotating s to the right by r means taking the suffix of length r and
        # attaching it in front. Equivalently, it is the same as a left rotation by (n - r).
        # Thus, t = s[n-r:] + s[:n-r] if and only if t is a rotation of s.
        #
        # A common trick is to check if t occurs in s+s. In particular, if t appears starting at
        # position pos in s+s (with pos < n) then t equals s left rotated by pos, and hence it also equals
        # s right rotated by (n - pos) mod n.
        #
        # Therefore, the set of valid net rotations is given by:
        #    validRS = { (n - pos) mod n   for each occurrence pos of t in s+s with pos < n }.
        #
        # Finally, the answer is the sum over all valid r:
        #    ans = (if 0 in validRS then T(0) else 0) + (|validRS| - [0 in validRS]) * T(nonzero)
        # (Remember to take the answer mod MOD.)
        
        # Step 1. Find all occurrences of t in s+s using KMP (only positions less than n)
        def kmp_search(text: str, pattern: str):
            m = len(pattern)
            lps = [0] * m  # longest proper prefix which is also suffix
            length = 0
            i = 1
            while i < m:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            occ = []
            i = 0  # index for text
            j = 0  # index for pattern
            N = len(text)
            while i < N:
                if text[i] == pattern[j]:
                    i += 1
                    j += 1
                    if j == m:
                        occ.append(i - j)
                        j = lps[j - 1]
                else:
                    if j != 0:
                        j = lps[j - 1]
                    else:
                        i += 1
            return occ
        
        doubled = s + s
        occ = kmp_search(doubled, t)
        valid_rs = set()
        for pos in occ:
            if pos < n:
                # t is s left rotated by pos, which is equivalent to a right rotation of (n - pos) mod n.
                r = (n - pos) % n
                valid_rs.add(r)
        
        count0 = 1 if 0 in valid_rs else 0
        count_nonzero = len(valid_rs) - count0
        
        # Step 2. Compute the number of move sequences that yield a given net rotation.
        A = pow(n - 1, k, MOD)  # (n-1)^k mod MOD
        B = 1 if k % 2 == 0 else MOD - 1  # (-1)^k mod MOD
        inv_n = pow(n, MOD - 2, MOD)  # Modular inverse of n mod MOD
        
        # For r = 0:
        T0 = (A + B * (n - 1)) % MOD
        T0 = (T0 * inv_n) % MOD
        # For any nonzero r:
        T_nonzero = (A - B) % MOD
        T_nonzero = (T_nonzero * inv_n) % MOD
        
        # Step 3. Sum the contributions of all valid rotations.
        ans = ( (T0 if count0 else 0) + count_nonzero * T_nonzero ) % MOD
        return ans

# The following code is provided for local testing purposes.
# It also uses a helper main routine that reads input and prints output.
if __name__=="__main__":
    import sys
    sol = Solution()
    data = sys.stdin.read().split()
    if data:
        s = data[0].strip()
        t = data[1].strip()
        k = int(data[2])
        print(sol.numberOfWays(s, t, k))
    else:
        # Example tests when no input is provided
        print(sol.numberOfWays("abcd", "cdab", 2))      # Expected output: 2
        print(sol.numberOfWays("ababab", "ababab", 1))    # Expected output: 2