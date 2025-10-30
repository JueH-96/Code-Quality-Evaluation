class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        n = len(word)
        max_len = 0
        left = 0
        
        for right in range(n):
            # Check all substrings ending at position right with length up to 10
            for length in range(1, min(11, right + 2)):
                start = right - length + 1
                if word[start:right + 1] in forbidden_set:
                    # Found a forbidden substring, move left to exclude it
                    left = max(left, start + 1)
            
            # Update the maximum length of valid substring
            max_len = max(max_len, right - left + 1)
        
        return max_len