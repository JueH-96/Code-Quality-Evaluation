class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden = set(forbidden)
        max_len = 0
        left = 0
        n = len(word)
        for right in range(n):
            # Check all possible forbidden substrings ending at 'right'
            for L in range(1, 11):
                s = right - L + 1
                if s < 0:
                    continue
                substring = word[s: right + 1]
                if substring in forbidden:
                    left = max(left, s + 1)
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        return max_len