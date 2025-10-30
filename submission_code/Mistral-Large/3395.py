class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter

        # Count the frequency of each character in the string s
        count = Counter(s)

        # Find the character with the maximum frequency
        max_freq = max(count.values())

        # The minimum length of t is the maximum frequency found
        return max_freq