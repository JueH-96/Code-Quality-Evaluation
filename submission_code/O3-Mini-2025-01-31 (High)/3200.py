MOD = 10**9 + 7

class Solution:
    def stringCount(self, n: int) -> int:
        mod = MOD
        # Explanation:
        # A string is "good" if it contains at least one 'l', at least two 'e',
        # and at least one 't'. Since we can rearrange the string arbitrarily,
        # the only requirement is that its multiset of characters contains
        # at least (l, e, e, t).
        #
        # Total # of strings of length n = 26^n.
        # Let:
        #   A = strings with no 'l'
        #   B = strings with no 't'
        #   C = strings with at most one 'e'
        #
        # Then, by inclusion–exclusion, valid strings count = 
        #   total - (|A| + |B| + |C|) + (|A∩B| + |A∩C| + |B∩C|) - |A∩B∩C|.
        #
        # We compute:
        #   |A| = 25^n
        #   |B| = 25^n
        #   |C| = (strings with 0 'e') + (strings with exactly 1 'e').
        #       With 0 'e': 25^n.
        #       With exactly 1 'e': choose 1 position for 'e', rest from the other 25 letters => n * 25^(n-1).
        #       So, |C| = 25^n + n * 25^(n-1).
        #
        #   |A∩B|: no 'l' and no 't' => allowed letters: 24 letters => 24^n.
        #   |A∩C|: no 'l' and at most one 'e'. 
        #         With 0 'e' (and no 'l'): allowed letters: 26 - {'l', 'e'} = 24 letters => 24^n.
        #         With exactly 1 'e': choose one position for 'e' and the rest from 24 letters => n * 24^(n-1).
        #         So, |A∩C| = 24^n + n * 24^(n-1).
        #   |B∩C|: no 't' and at most one 'e'. This count is the same as |A∩C|:
        #         |B∩C| = 24^n + n * 24^(n-1).
        #   |A∩B∩C|: no 'l', no 't', and at most one 'e'.
        #         For 0 'e': allowed letters: 26 - {l, t, e} = 23 letters => 23^n.
        #         For exactly 1 'e': choose one spot and the others from 23 letters => n * 23^(n-1).
        #         So, |A∩B∩C| = 23^n + n * 23^(n-1).
        #
        # Then the number of good strings is:
        #
        #    good = 26^n
        #           - [|A| + |B| + |C|]
        #           + [|A∩B| + |A∩C| + |B∩C|]
        #           - |A∩B∩C|
        #
        # And we take the answer modulo 10^9+7.
        
        total = pow(26, n, mod)
        
        # Count strings missing 'l'
        count_no_l = pow(25, n, mod)
        # Count strings missing 't'
        count_no_t = pow(25, n, mod)
        
        # Count strings with at most one 'e'
        count_zero_e = pow(25, n, mod)
        count_one_e = (n * pow(25, n - 1, mod)) % mod
        count_at_most_one_e = (count_zero_e + count_one_e) % mod
        
        # Intersection: no 'l' and no 't'
        count_no_l_no_t = pow(24, n, mod)
        
        # Intersection: no 'l' and at most one 'e'
        count_no_l_zero_e = pow(24, n, mod)
        count_no_l_one_e = (n * pow(24, n - 1, mod)) % mod
        count_no_l_at_most_one_e = (count_no_l_zero_e + count_no_l_one_e) % mod
        
        # Intersection: no 't' and at most one 'e'
        count_no_t_zero_e = pow(24, n, mod)
        count_no_t_one_e = (n * pow(24, n - 1, mod)) % mod
        count_no_t_at_most_one_e = (count_no_t_zero_e + count_no_t_one_e) % mod
        
        # Intersection: no 'l', no 't', and at most one 'e'
        count_no_l_t_zero_e = pow(23, n, mod)
        count_no_l_t_one_e = (n * pow(23, n - 1, mod)) % mod
        count_no_l_t_at_most_one_e = (count_no_l_t_zero_e + count_no_l_t_one_e) % mod
        
        # Use Inclusion-Exclusion to count invalid ("bad") strings:
        bad = (count_no_l + count_no_t + count_at_most_one_e) % mod
        bad = (bad - count_no_l_no_t - count_no_l_at_most_one_e - count_no_t_at_most_one_e) % mod
        bad = (bad + count_no_l_t_at_most_one_e) % mod
        
        good = (total - bad) % mod
        return good

# You can run some basic tests
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.stringCount(4))    # Expected output: 12
    # Example 2:
    print(sol.stringCount(10))   # Expected output: 83943898