class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        n = len(word)
        left = 0
        max_len = 0
        
        for right in range(n):
            # Check if any forbidden string ends at position right
            for length in range(1, 11):  # max length of forbidden is 10
                start = right - length + 1
                if start >= 0:
                    substring = word[start:right + 1]
                    if substring in forbidden_set:
                        left = max(left, start + 1)  # Move left to exclude this forbidden substring
            
            max_len = max(max_len, right - left + 1)
        
        return max_len