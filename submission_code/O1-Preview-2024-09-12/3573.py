class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter, defaultdict

        counts_word2 = Counter(word2)
        required = len(counts_word2)
        counts_window = defaultdict(int)
        formed = 0
        left = 0
        total_substrings = 0
        n = len(word1)

        for right in range(n):
            char = word1[right]
            counts_window[char] +=1
            if char in counts_word2 and counts_window[char] == counts_word2[char]:
                formed +=1

            while left <= right and formed == required:
                # When formed == required, we have a valid window
                # All substrings ending at 'right' and starting from 'left' to 'right' are valid
                total_substrings += n - right  # Since rearrangement is allowed, we can use the same counts

                # Shrink the window from the left
                counts_window[word1[left]] -=1
                if word1[left] in counts_word2 and counts_window[word1[left]] < counts_word2[word1[left]]:
                    formed -=1
                left +=1

        return total_substrings