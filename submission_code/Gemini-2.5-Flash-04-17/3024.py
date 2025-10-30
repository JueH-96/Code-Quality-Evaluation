import sys

MOD = 10**9 + 7

# Function to compute modular exponentiation (a^b % m)
def power(a, b, m):
    a %= m
    res = 1
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % m
        a = (a * a) % m
        b //= 2
    return res

# Function to compute modular inverse (a^-1 % m) using Fermat's Little Theorem
# m must be prime, and a must not be a multiple of m.
def modInverse(a, m):
    return power(a, m - 2, m)

class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)

        # KMP prefix function calculation
        # pi[i] is the length of the longest proper prefix of pattern[:i+1]
        # which is also a suffix of pattern[:i+1].
        def compute_pi(pattern):
            m = len(pattern)
            pi = [0] * m
            j = 0 # length of the previous longest prefix suffix
            for i in range(1, m):
                while j > 0 and pattern[i] != pattern[j]:
                    j = pi[j - 1]
                if pattern[i] == pattern[j]:
                    j += 1
                pi[i] = j
            return pi

        # KMP search for all occurrences of pattern in text
        # Finds starting indices of occurrences.
        def find_all_occurrences(text, pattern, pi):
            n_text = len(text)
            m_pattern = len(pattern)
            occurrences = []
            j = 0 # length of current match
            for i in range(n_text):
                while j > 0 and text[i] != pattern[j]:
                    j = pi[j - 1]
                if text[i] == pattern[j]:
                    j += 1
                if j == m_pattern:
                    # Match found ending at index i in text
                    start_index = i - m_pattern + 1
                    occurrences.append(start_index)
                    # Continue search for more matches starting after this one.
                    # The next state corresponds to matching the longest proper prefix of pattern
                    # that is also a suffix of the current match (which is the whole pattern).
                    j = pi[m_pattern - 1] # Since j == m_pattern

            return occurrences

        # Find the set of required shifts p such that s shifted by p equals t.
        # A left shift by p means the string becomes s[p:] + s[:p].
        # This is equivalent to finding occurrences of t in s + s.
        # The starting index of t in s + s is the shift amount p.
        # We are interested in shifts p in the range [0, n-1].
        double_s = s + s
        pi_t = compute_pi(t)
        all_occurrences = find_all_occurrences(double_s, t, pi_t)

        shifts_S = set()
        for p in all_occurrences:
            # An occurrence starting at index p in s+s means s+s[p : p+n] == t.
            # If 0 <= p < n, this corresponds to a left shift by p. Add p to the set.
            # Occurrences starting at p >= n also correspond to cyclic shifts.
            # An occurrence of t starting at index `p` in `s+s` means the string `s` shifted left by `p` equals `t`.
            # We need to find all distinct values `p mod n` such that `s` shifted by `p` equals `t`.
            # The starting indices of occurrences of `t` in `s+s` precisely give the shift amounts `p` such that `s` shifted by `p` equals `t`.
            # We need the set of unique required shifts modulo n.
            # Since KMP finds all occurrences, we just need the starting indices less than n.
            # Any occurrence starting at index p in s+s where p < n gives a unique shift p.
            if p < n:
                 shifts_S.add(p)


        # If t is not a cyclic shift of s (i.e. no occurrence starting at index < n in s+s), return 0
        if not shifts_S:
             return 0

        # Calculate terms for the formulas dp[k][p]
        # dp[i][0] = ((n-1)^i + (n-1)(-1)^i) / n
        # dp[i][j] = ((n-1)^i - (-1)^i) / n for j != 0

        N_minus_1 = n - 1
        
        # n >= 2, so n-1 >= 1. power(n-1, k, MOD) is fine.
        N_minus_1_pow_k = power(N_minus_1, k, MOD)

        # Calculate (-1)^k mod MOD
        minus_one_pow_k = 1 if k % 2 == 0 else MOD - 1

        # Calculate modular inverse of n
        # n >= 2 and MOD is prime, so n and MOD are coprime.
        n_inv = modInverse(n, MOD)

        total_ways = 0

        for p in shifts_S:
            if p == 0:
                # dp[k][0] = ((n-1)^k + (n-1)(-1)^k) / n
                term1 = N_minus_1_pow_k
                term2 = (N_minus_1 * minus_one_pow_k) % MOD
                
                numerator = (term1 + term2) % MOD
                ways = (numerator * n_inv) % MOD
            else:
                # dp[k][p] = ((n-1)^k - (-1)^k) / n for p != 0
                term1 = N_minus_1_pow_k
                term2 = minus_one_pow_k

                numerator = (term1 - term2) % MOD
                numerator = (numerator + MOD) % MOD # Ensure positive result

                ways = (numerator * n_inv) % MOD

            total_ways = (total_ways + ways) % MOD

        return total_ways