from collections import Counter

class Solution:
    def maxDifference(self, s: str) -> int:
        """
        Finds the maximum difference between the frequency of a character
        with an odd frequency and a character with an even frequency.

        Args:
            s: The input string consisting of lowercase English letters.

        Returns:
            The maximum difference (odd frequency - even frequency).
        """
        # 1. Calculate the frequency of each character
        freq_counts = Counter(s)

        # 2. Separate frequencies into two lists: odd and even
        odd_frequencies = []
        even_frequencies = []
        for freq in freq_counts.values():
            if freq % 2 == 1:
                odd_frequencies.append(freq)
            else: # freq % 2 == 0
                even_frequencies.append(freq)

        # 3. To maximize (odd frequency - even frequency), we need the
        # largest odd frequency and the smallest even frequency.
        # The constraints guarantee that s contains at least one character
        # with an odd frequency and one with an even frequency, so
        # odd_frequencies and even_frequencies lists are not empty.
        max_odd_freq = max(odd_frequencies)
        min_even_freq = min(even_frequencies)

        # 4. Calculate the maximum difference
        max_diff = max_odd_freq - min_even_freq

        return max_diff