class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        n = len(word)
        max_len = 0
        left = 0
        for right in range(n):
            for i in range(right, max(right - 10, left - 1), -1):
                if word[i:right + 1] in forbidden_set:
                    left = i + 1
                    break
            max_len = max(max_len, right - left + 1)
        return max_len