import math

class Solution:
    def _get_k_reduced(self, k: int) -> int:
        """
        Calculates k_reduced such that (p*p) % k == 0 iff p % k_reduced == 0.
        This holds for p >= 1. The main loop ensures p >= 1 for beautiful substrings.
        """
        if k == 1:
            return 1
        
        k_reduced = 1
        temp_k = k
        
        # Factor out powers of 2
        if temp_k % 2 == 0:
            count = 0
            while temp_k % 2 == 0:
                temp_k //= 2
                count += 1
            # For a prime factor q_i with exponent a_i in k (k = ... * q_i^a_i * ...)
            # q_i must appear in p with an exponent of at least ceil(a_i / 2).
            exponent_in_k_reduced = (count + 1) // 2 # Integer arithmetic for ceil(count / 2)
            k_reduced *= (2 ** exponent_in_k_reduced)

        # Factor out powers of odd primes
        d = 3
        while d * d <= temp_k: # Iterate up to sqrt of the remaining temp_k
            if temp_k % d == 0:
                count = 0
                while temp_k % d == 0:
                    temp_k //= d
                    count += 1
                
                exponent_in_k_reduced = (count + 1) // 2
                k_reduced *= (d ** exponent_in_k_reduced)
            d += 2 # Check only odd factors
        
        if temp_k > 1: # Remaining temp_k is a prime factor itself
            # This prime factor has exponent 1 in the remaining temp_k.
            # count = 1 for this prime factor.
            exponent_in_k_reduced = 1 # ceil(1 / 2) = 1
            k_reduced *= (temp_k ** exponent_in_k_reduced) # temp_k is the prime factor
            
        return k_reduced

    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        vowels_set = {'a', 'e', 'i', 'o', 'u'}
        
        # Calculate k_reduced for the optimized condition p % k_reduced == 0.
        # This is equivalent to (p*p) % k == 0 for p >= 1.
        mod_check_value = self._get_k_reduced(k)

        beautiful_count = 0
        for i in range(n):
            current_vowels = 0
            current_consonants = 0
            for j in range(i, n):
                char = s[j]
                if char in vowels_set:
                    current_vowels += 1
                else:
                    current_consonants += 1
                
                # Check condition 1: vowels == consonants
                if current_vowels == current_consonants:
                    # Let p = current_vowels.
                    # Substring s[i..j] has length L = j-i+1. Since j >= i, L >= 1.
                    # If current_vowels == current_consonants == p, then L = 2*p.
                    # So, 2*p >= 1, which means p >= 1/2.
                    # Since p is an integer (count of vowels), p must be >= 1.
                    # This correctly ensures we consider non-empty strings resulting in p > 0.
                    
                    # Check condition 2 (optimized form): p % k_reduced == 0
                    if current_vowels % mod_check_value == 0:
                        beautiful_count += 1
                        
        return beautiful_count