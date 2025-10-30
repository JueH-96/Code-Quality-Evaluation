from collections import Counter

class Solution:
    def findValidPair(self, s: str) -> str:
        # Pre–compute the frequency of every digit in the string
        freq = Counter(s)

        # Scan the string from left to right, looking at every adjacent pair
        for i in range(len(s) - 1):
            first, second = s[i], s[i + 1]

            # The digits must be different
            if first == second:
                continue

            # Each digit in the pair must appear in the string exactly
            # as many times as its numeric value
            if freq[first] == int(first) and freq[second] == int(second):
                return first + second

        # No valid pair found
        return ""