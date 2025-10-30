class Solution:
    def maxDifference(self, s: str) -> int:
        import collections

        # Step 1: Count the frequency of each character. The .values() method
        # provides an iterable view of the frequencies.
        frequencies = collections.Counter(s).values()
        
        # Step 2: Use list comprehensions to partition the frequencies into two
        # lists based on their parity.
        # The problem guarantees at least one odd and one even frequency exist.
        odd_freqs = [f for f in frequencies if f % 2 != 0]
        even_freqs = [f for f in frequencies if f % 2 == 0]
        
        # Step 3: To maximize the difference (odd_freq - even_freq), we must find
        # the maximum odd frequency and the minimum even frequency.
        max_odd = max(odd_freqs)
        min_even = min(even_freqs)
        
        # Step 4: Return the calculated maximum difference.
        return max_odd - min_even