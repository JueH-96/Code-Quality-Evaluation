class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter

        # Count the frequency of each character in the word
        freq = Counter(word)

        # Find the minimum and maximum frequency
        min_freq = min(freq.values())
        max_freq = max(freq.values())

        # If the difference between max and min frequency is within k, no deletions are needed
        if max_freq - min_freq <= k:
            return 0

        # Otherwise, we need to delete characters to make the frequencies within k of each other
        deletions = 0
        for count in freq.values():
            if count > min_freq + k:
                deletions += count - (min_freq + k)

        return deletions