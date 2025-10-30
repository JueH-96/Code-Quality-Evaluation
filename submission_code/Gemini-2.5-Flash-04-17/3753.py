from collections import Counter

class Solution:
    def maxDifference(self, s: str) -> int:
        """
        Finds the maximum difference between the frequency of a character with
        an odd frequency and a character with an even frequency.

        Args:
            s: The input string consisting of lowercase English letters.

        Returns:
            The maximum difference (odd frequency - even frequency).
        """
        freq_counter = Counter(s)
        
        odd_freqs = []
        even_freqs = []
        
        for freq in freq_counter.values():
            if freq % 2 == 0:
                even_freqs.append(freq)
            else:
                odd_freqs.append(freq)
        
        # The problem guarantees there is at least one odd and one even frequency,
        # so odd_freqs and even_freqs will not be empty.
        
        max_odd_freq = max(odd_freqs)
        min_even_freq = min(even_freqs)
        
        return max_odd_freq - min_even_freq