class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        n = len(word)
        forbidden_set = set(forbidden)
        max_len = 0
        
        for i in range(n):
            for j in range(i + 1, n + 1):
                substring = word[i:j]
                if any(f in substring for f in forbidden_set):
                    break
                max_len = max(max_len, j - i)
        
        return max_len