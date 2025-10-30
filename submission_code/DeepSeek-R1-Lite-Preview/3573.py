class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import defaultdict

        # Frequency counts for word2
        target_counts = defaultdict(int)
        for char in word2:
            target_counts[char] += 1

        unique_chars = len(target_counts)
        have = 0

        # Current frequency counts in the window
        current_counts = defaultdict(int)

        result = 0
        left = 0
        n = len(word1)

        for right in range(n):
            char = word1[right]
            if char in target_counts:
                current_counts[char] += 1
                if current_counts[char] == target_counts[char]:
                    have += 1

            # When the current window is valid
            while have == unique_chars:
                # All substrings from left to right are valid
                result += n - right
                # Move left to find the next valid window
                left_char = word1[left]
                if left_char in target_counts:
                    current_counts[left_char] -= 1
                    if current_counts[left_char] < target_counts[left_char]:
                        have -= 1
                left += 1

        return result