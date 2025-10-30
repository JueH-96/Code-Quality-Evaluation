import collections

class Solution:
    def maxDifference(self, s: str) -> int:
        # Calculate frequencies of each character
        freq_counter = collections.Counter(s)

        # Separate frequencies into odd and even lists
        odd_freqs = []
        even_freqs = []

        for freq in freq_counter.values():
            if freq % 2 == 1:
                odd_freqs.append(freq)
            else: # freq % 2 == 0
                even_freqs.append(freq)

        # The problem guarantees there is at least one character with an odd frequency
        # and one with an even frequency, so odd_freqs and even_freqs will not be empty.

        # Find the maximum frequency among odd frequencies
        max_odd = max(odd_freqs)

        # Find the minimum frequency among even frequencies
        min_even = min(even_freqs)

        # Calculate the maximum difference
        return max_odd - min_even