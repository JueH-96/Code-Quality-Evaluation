class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_len = 0
        left = 0
        n = len(word)
        for right in range(n):
            for l in range(1, min(11, right - left + 2)):
                substring = word[right - l + 1: right + 1]
                if substring in forbidden_set:
                    left = right - l + 2
                    break
            max_len = max(max_len, right - left + 1)
        return max_len