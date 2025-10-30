class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_forbidden_length = max(len(f) for f in forbidden)
        max_length = 0
        n = len(word)
        
        left = 0
        for right in range(n):
            # Check if any suffix ending at position right is forbidden
            # We only need to check suffixes up to max_forbidden_length
            for i in range(right, max(right - max_forbidden_length, left - 1), -1):
                if word[i:right+1] in forbidden_set:
                    left = i + 1
                    break
            max_length = max(max_length, right - left + 1)
        
        return max_length