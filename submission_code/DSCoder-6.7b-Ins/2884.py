class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        n = len(word)
        max_len = 0
        forbidden_set = set(forbidden)
        for start in range(n):
            for end in range(start+1, n+1):
                substring = word[start:end]
                if not any(substring.endswith(f) for f in forbidden_set):
                    max_len = max(max_len, len(substring))
        return max_len