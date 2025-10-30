class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter

        # Count the frequency of each character in the string
        freq = Counter(s)

        # Find the maximum frequency
        max_freq = max(freq.values())

        # Calculate the total number of operations needed
        total_operations = 0
        for count in freq.values():
            total_operations += max_freq - count

        return total_operations