class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        n = len(word)
        left = 0
        max_len = 0
        
        for right in range(n):
            # Check for forbidden substrings ending at position 'right'
            # We only need to check substrings of length up to 10 (max forbidden length)
            for start in range(max(0, right - 9), right + 1):
                substring = word[start:right + 1]
                if substring in forbidden_set:
                    # Move left pointer to avoid this forbidden substring
                    left = max(left, start + 1)
                    break
            
            # Update maximum length
            max_len = max(max_len, right - left + 1)
        
        return max_len