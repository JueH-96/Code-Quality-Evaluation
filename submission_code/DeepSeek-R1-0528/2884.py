class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        left = 0
        max_len = 0
        n = len(word)
        for right in range(n):
            start = max(left, right - 9)
            for i in range(right, start - 1, -1):
                if word[i:right+1] in forbidden_set:
                    left = i + 1
                    break
            max_len = max(max_len, right - left + 1)
        return max_len