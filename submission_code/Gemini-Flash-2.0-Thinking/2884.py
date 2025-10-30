class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        n = len(word)
        forbidden_set = set(forbidden)
        max_len = 0
        start = 0
        for r in range(n):
            for f in forbidden_set:
                # Check if f is a substring of word[start:r+1]
                try:
                    index = word.index(f, start, r + 1)
                    start = max(start, index + len(f))
                except ValueError:
                    pass
            max_len = max(max_len, r - start + 1)
        return max_len