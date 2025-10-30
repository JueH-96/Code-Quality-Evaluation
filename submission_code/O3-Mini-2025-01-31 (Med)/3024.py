class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        mod = 10**9 + 7
        n = len(s)
        
        # First, we need to count the number of rotations j (0 <= j < n)
        # such that rotating s by j gives t. A rotation by j means:
        #    s_rot = s[j:] + s[:j]
        # Note that the allowed operation – removing a non‐empty proper suffix and
        # putting it at the front – is equivalent (when viewed modulo n) to adding
        # a rotation amount in {1,2,..., n-1}. Therefore, every operation chooses an
        # r (with r in {1, …, n-1}) and the net effect after k operations is a rotation by
        # R = (r1 + r2 + … + rk) mod n.
        #
        # We first count the number (m) of indices j with 0 <= j < n such that rotating s by j equals t.
        # (Notice: if s is periodic, it might be that more than one rotation equals t.)
        # We can do this using the following observation:
        #    t is a rotation of s  if and only if t is a substring of s+s (and starting index < n).
        # Standard KMP is used to count occurrences.
        def count_occurrences(text, pattern):
            m = len(pattern)
            pi = [0]*m
            j = 0
            # prefix (failure) function for pattern
            for i in range(1, m):
                while j > 0 and pattern[i] != pattern[j]:
                    j = pi[j-1]
                if pattern[i] == pattern[j]:
                    j += 1
                    pi[i] = j
            
            occ = 0
            j = 0
            # search pattern in text (consider indices < n only)
            for i in range(len(text)):
                while j > 0 and text[i] != pattern[j]:
                    j = pi[j-1]
                if text[i] == pattern[j]:
                    j += 1
                    if j == m:
                        # occurrence found starting at index (i - m + 1)
                        if i - m + 1 < n:
                            occ += 1
                        j = pi[j-1]
            return occ
        
        full_text = s + s
        m_occ = count_occurrences(full_text, t)
        if m_occ == 0:
            # t is not a rotation of s so it's impossible.
            return 0
        
        # Our allowed operation adds a rotation in {1, 2, ..., n-1}.
        # Let P(x) = x^1 + x^2 + ... + x^(n-1).
        # Then, after k operations, the number of sequences that produce a net rotation R (mod n)
        # is the coefficient of x^R in P(x)^k.
        #
        # A standard trick using roots of unity shows that
        #   f(R) = (1/n) * [ (n-1)^k + { (n-1)*(-1)^k if R = 0, or -(-1)^k if R != 0 } ]
        #
        # Hence, if we let:
        #   f(0) = 1/n * [ (n-1)^k + (n-1)*(-1)^k ]     when R == 0, and
        #   f(j) = 1/n * [ (n-1)^k - (-1)^k ]               for any nonzero residue j.
        #
        # Now let R be the set of rotations that send s to t.
        # One special rotation is j = 0 (i.e. s == t) if s equals t.
        # Let:
        #   m = |R|, and let δ = 1 if 0 ∈ R (i.e. if s == t), else 0.
        #
        # Then the answer (number of sequences that produce t) is:
        #   if s == t (i.e. δ = 1):
        #      answer = f(0) + (m - 1) * f(nonzero)
        #             = 1/n * [ m*(n-1)^k + (n-m)*(-1)^k ]
        #   if s != t (δ = 0):
        #      answer = m * f(nonzero)
        #             = 1/n * [ m*((n-1)^k - (-1)^k) ]
        #
        # Compute these values modulo mod.
        
        inv_n = pow(n, mod-2, mod)
        pow_val = pow(n-1, k, mod)
        # pow(-1, k, mod) correctly computes (-1)^k modulo mod.
        pow_neg1 = pow(-1, k, mod)
        
        # δ is 1 if rotation 0 yields t, i.e., if s == t.
        zero_in = (s == t)
        if zero_in:
            # m_occ includes index 0 since s == t.
            res = (m_occ * pow_val + (n - m_occ) * pow_neg1) % mod
        else:
            res = (m_occ * ((pow_val - pow_neg1) % mod)) % mod
        
        return (res * inv_n) % mod

# -------------------------
# For testing locally:
if __name__ == '__main__':
    sol = Solution()
    
    # Example 1:
    s1 = "abcd"
    t1 = "cdab"
    k1 = 2
    print(sol.numberOfWays(s1, t1, k1))  # Expected output: 2

    # Example 2:
    s2 = "ababab"
    t2 = "ababab"
    k2 = 1
    print(sol.numberOfWays(s2, t2, k2))  # Expected output: 2