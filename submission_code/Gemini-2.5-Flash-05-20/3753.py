import collections

class Solution:
    def maxDifference(self, s: str) -> int:
        # Step 1: Calculate frequencies of all characters in the string s.
        # collections.Counter is efficient for this purpose.
        char_counts = collections.Counter(s)

        # Step 2: Initialize lists to store frequencies based on their parity (odd/even).
        odd_frequencies = []
        even_frequencies = []

        # Iterate through the frequencies (values) obtained from the Counter.
        for freq in char_counts.values():
            if freq % 2 == 1:  # If the frequency is odd
                odd_frequencies.append(freq)
            else:  # If the frequency is even
                even_frequencies.append(freq)

        # Step 3: Find the maximum frequency among characters with odd frequencies.
        # The problem constraints guarantee that 'odd_frequencies' will not be empty.
        max_odd_freq = max(odd_frequencies)

        # Step 4: Find the minimum frequency among characters with even frequencies.
        # The problem constraints guarantee that 'even_frequencies' will not be empty.
        min_even_freq = min(even_frequencies)

        # Step 5: Calculate and return the maximum difference.
        # This is (maximum odd frequency) - (minimum even frequency).
        return max_odd_freq - min_even_freq