class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import defaultdict

        if not word2 or not word1:
            return 0

        need = [0] * 26
        for c in word2:
            need[ord(c) - ord('a')] += 1

        required = 0
        for cnt in need:
            if cnt > 0:
                required += 1

        window = [0] * 26
        formed = 0
        left = 0
        count = 0
        n = len(word1)

        for right in range(n):
            char_idx = ord(word1[right]) - ord('a')
            window[char_idx] += 1

            if need[char_idx] > 0 and window[char_idx] == need[char_idx]:
                formed += 1

            while left <= right and formed == required:
                count += n - right

                left_char_idx = ord(word1[left]) - ord('a')
                window[left_char_idx] -= 1
                if need[left_char_idx] > 0 and window[left_char_idx] < need[left_char_idx]:
                    formed -= 1
                left += 1

        return count