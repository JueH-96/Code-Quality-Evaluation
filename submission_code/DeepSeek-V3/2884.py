class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_len = 0
        left = 0
        n = len(word)
        # Precompute the maximum length of forbidden words
        max_forbidden_len = max(len(s) for s in forbidden) if forbidden else 0
        
        for right in range(n):
            # Check all possible substrings ending at 'right' with length up to max_forbidden_len
            for l in range(1, min(max_forbidden_len, right - left + 1) + 1):
                substring = word[right - l + 1: right + 1]
                if substring in forbidden_set:
                    left = right - l + 2
                    break
            # Update the maximum length
            max_len = max(max_len, right - left + 1)
        
        return max_len