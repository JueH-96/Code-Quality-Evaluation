class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        # Calculate the total number of strings of length n
        total_strings = 26**n
        
        # Calculate the number of strings that do not contain 'l', 'e', 'e', 't'
        no_l = 25**n
        no_e = 24**n
        no_t = 25**n
        no_le = 24**n
        no_lt = 24**n
        no_et = 24**n
        no_let = 23**n
        no_leet = 23**n
        
        # Apply the principle of inclusion-exclusion
        bad_strings = no_l + no_e + no_t - no_le - no_lt - no_et + no_let + no_leet
        
        # Calculate the number of good strings
        good_strings = total_strings - bad_strings
        
        # Return the result modulo MOD
        return good_strings % MOD