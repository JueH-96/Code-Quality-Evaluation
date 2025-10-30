import math

class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        vowels_set = {'a', 'e', 'i', 'o', 'u'}

        # Calculate k_prime such that m^2 % k == 0 <=> m % k_prime == 0
        # where m is the number of vowels (and consonants) in the substring.
        # k_prime is derived from k by taking ceil(exponent/2) for each prime factor.
        k_prime = 1
        temp_k = k
        p = 2
        # Iterate up to sqrt(temp_k) to find prime factors
        while p * p <= temp_k:
            if temp_k % p == 0:
                count = 0
                # Count the exponent of prime factor p in temp_k
                while temp_k % p == 0:
                    count += 1
                    temp_k //= p
                # Include p raised to the power ceil(count/2) in k_prime
                k_prime *= (p ** ((count + 1) // 2))
            # Check next potential prime factor
            p += 1
        # If temp_k > 1 after the loop, the remaining temp_k is a prime factor itself
        if temp_k > 1: 
            # The exponent of this prime factor is 1, ceil(1/2) = 1
            k_prime *= temp_k
            
        # Conditions for a beautiful substring s[i:j] (0-indexed, inclusive i to j)
        # 1. vowels == consonants.
        #    Let prefix_sum[p] be the sum of (+1 for vowel, -1 for consonant) for s[0:p-1].
        #    prefix_sum[0] = 0. prefix_sum[p+1] = prefix_sum[p] + (1 if s[p] is vowel else -1).
        #    The sum for substring s[i:j] is prefix_sum[j+1] - prefix_sum[i].
        #    vowels == consonants implies this sum is 0: prefix_sum[j+1] == prefix_sum[i].
        # 2. (vowels * consonants) % k == 0. If vowels == consonants = m, this is m^2 % k == 0,
        #    which is equivalent to m % k_prime == 0.
        #    Also, vowels + consonants = length = j - i + 1.
        #    If vowels == consonants = m, then length = 2 * m.
        #    So m = length / 2 = (j - i + 1) / 2. This requires length to be even.
        #    The condition becomes ((j - i + 1) / 2) % k_prime == 0.
        # Combined conditions for substring s[i:j] (0 <= i <= j < n):
        # prefix_sum[j+1] == prefix_sum[i]
        # AND (j - i + 1) is even
        # AND ((j - i + 1) / 2) % k_prime == 0.
        # The length condition (j - i + 1) is even and ((j - i + 1) / 2) is a multiple of k_prime,
        # means (j - i + 1) must be a multiple of 2 * k_prime.
        # Let offset = 2 * k_prime. We need (j - i + 1) % offset == 0.
        # This implies (j + 1) % offset == i % offset.

        offset = 2 * k_prime

        # Calculate prefix sum array (+1 for vowel, -1 for consonant)
        # prefix_sum[p] is the sum for s[0:p] (exclusive end index, 0-indexed string)
        # prefix_sum array has length n + 1. prefix_sum[0] = 0.
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i]
            if s[i] in vowels_set:
                prefix_sum[i+1] += 1
            else:
                prefix_sum[i+1] -= 1

        count = 0
        # Map to store counts of (prefix_sum_value, index_modulo_offset)
        # Key: (prefix_sum[p], p % offset) where p is an index in the prefix_sum array (0 to n).
        # Value: count of times this pair has been seen.
        counts_map = {}

        # The starting point of a substring s[i:j] corresponds to index i in prefix_sum.
        # The ending point corresponds to index j+1 in prefix_sum.
        # We iterate the *end* index of the prefix_sum array, from 1 to n.
        # Let `end_ps_idx` be the index in prefix_sum (1 to n). It corresponds to character s[end_ps_idx - 1].
        # We look for a `start_ps_idx` (0 to end_ps_idx - 1).
        # Substring s[start_ps_idx : end_ps_idx - 1] (0-indexed string).
        # Conditions translated to prefix_sum indices:
        # 1. prefix_sum[end_ps_idx] == prefix_sum[start_ps_idx]
        # 2. (end_ps_idx - start_ps_idx) is a multiple of offset = 2 * k_prime.
        # This implies end_ps_idx % offset == start_ps_idx % offset.

        # Initialize map with the state at prefix_sum index 0.
        # This index 0 is a potential start_ps_idx for substrings starting at s[0].
        # prefix_sum[0] is the value before processing any character.
        counts_map[(prefix_sum[0], 0 % offset)] = 1

        # Iterate through all possible ending indices in the prefix_sum array (from 1 to n)
        # `end_ps_idx` represents the point *after* the substring ends.
        # The actual 0-indexed substring ends at `end_ps_idx - 1`.
        for end_ps_idx in range(1, n + 1):
            current_ps = prefix_sum[end_ps_idx]
            current_rem = end_ps_idx % offset

            # We look for a previous index `start_ps_idx` (< end_ps_idx) such that
            # prefix_sum[start_ps_idx] == current_ps AND start_ps_idx % offset == current_rem.
            # Any such `start_ps_idx` forms a beautiful substring ending at index `end_ps_idx - 1`
            # and starting at index `start_ps_idx`.
            # The number of such previous `start_ps_idx` values is stored in the map.
            
            match_key = (current_ps, current_rem)
            count += counts_map.get(match_key, 0)

            # Add the current index `end_ps_idx` state to the map, so it can be used as
            # a `start_ps_idx` for future ending indices.
            counts_map[match_key] = counts_map.get(match_key, 0) + 1

        return count