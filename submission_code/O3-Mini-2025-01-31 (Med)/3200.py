MOD = 10**9+7

class Solution:
    def stringCount(self, n: int) -> int:
        # If n < 4, then it's impossible for any string of length n to even have the characters "l", "e", "e", "t".
        if n < 4:
            return 0

        # The key observation is that a string s of length n can be rearranged to have "leet" as a contiguous substring
        # if and only if s has at least one 'l', at least two 'e's, and at least one 't'.
        # In other words, s must have:
        #   count[l] >= 1, count[e] >= 2, count[t] >= 1.
        #
        # Total number of strings of length n is: 26^n.
        # We now use Inclusion-Exclusion to subtract strings that are missing one or more of our required counts.
        #
        # Let:
        #   A = set of strings that do not contain 'l'.           Count = 25^n.
        #   Z = set of strings that do not contain 't'.           Count = 25^n.
        #   B = set of strings that do not have at least 2 'e's,
        #       i.e. strings having 0 or 1 occurrence of 'e'.
        #       Count0 = strings with 0 e = 25^n, because each position can be any of the 25 other letters.
        #       Count1 = strings with exactly 1 e = n * 25^(n-1). 
        #       So, total for B = 25^n + n * 25^(n-1).
        #
        # Next we compute the pairwise intersections:
        #   A ∩ B: strings with no 'l' and with fewer than 2 'e's. Now the allowed letters come from 26 removed 'l' → 25 letters.
        #           For 0 e, we choose from 24 letters (all but 'l' and 'e'): 24^n.
        #           For exactly 1 e, choose 1 position for 'e' and fill the rest with 24 letters: n * 24^(n-1).
        #           So Count(A ∩ B) = 24^n + n * 24^(n-1).
        #
        #   A ∩ Z: strings with no 'l' and no 't'. Allowed alphabet has 24 letters, so Count = 24^n.
        #
        #   B ∩ Z: strings with no 't' and fewer than 2 'e's. Now the allowed alphabet is 26 - 1 = 25 letters.
        #           For 0 e, choose from 24 letters (all but e and t): 24^n.
        #           For exactly 1 e, choose one position for e and fill the rest with 24 letters: n * 24^(n-1).
        #           So Count(B ∩ Z) = 24^n + n * 24^(n-1).
        #
        # And finally:
        #   A ∩ B ∩ Z: strings with no 'l', no 't', and with fewer than 2 'e's.
        #               Allowed letters: 26 - {l, t} = 24 letters.
        #               For 0 e, choose from 23 letters (exclude e as well): 23^n.
        #               For exactly 1 e, choose one position for e and fill the rest with 23 letters: n * 23^(n-1).
        #               So Count = 23^n + n * 23^(n-1).
        #
        # Using inclusion-exclusion, the number of good strings is:
        #   Good = Total - (|A| + |B| + |Z|)
        #          + (|A ∩ B| + |A ∩ Z| + |B ∩ Z|)
        #          - |A ∩ B ∩ Z|
        #
        # Plugging in the numbers, we have:
        #
        #   Good = 26^n
        #          - [25^n + (25^n + n * 25^(n-1)) + 25^n]
        #          + [(24^n + n * 24^(n-1)) + 24^n + (24^n + n * 24^(n-1))]
        #          - [23^n + n * 23^(n-1)]
        #
        # Simplify:
        #
        #   Good = 26^n - (3 * 25^n + n * 25^(n-1))
        #          + (3 * 24^n + 2 * n * 24^(n-1))
        #          - (23^n + n * 23^(n-1))
        #
        # We perform all calculations modulo 10^9+7.
        
        mod = MOD
        
        term1 = pow(26, n, mod)
        term2 = (3 * pow(25, n, mod) + n * pow(25, n-1, mod)) % mod
        term3 = (3 * pow(24, n, mod) + 2 * n * pow(24, n-1, mod)) % mod
        term4 = (pow(23, n, mod) + n * pow(23, n-1, mod)) % mod
        
        result = (term1 - term2 + term3 - term4) % mod
        return result

# Sample testing
if __name__ == "__main__":
    sol = Solution()
    # Example 1: n = 4
    print(sol.stringCount(4))   # Expected output: 12
    # Example 2: n = 10
    print(sol.stringCount(10))  # Expected output: 83943898