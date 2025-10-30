class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # Convert forbidden list to set for O(1) lookup
        forbidden_set = set(forbidden)
        
        # Find the maximum length of forbidden strings
        max_forbidden_len = max(len(f) for f in forbidden) if forbidden else 0
        
        n = len(word)
        left = 0
        max_length = 0
        
        # Iterate with right pointer
        for right in range(n):
            # Check substrings ending at right position
            # We only need to check substrings of length up to max_forbidden_len
            for start in range(max(left, right - max_forbidden_len + 1), right + 1):
                if word[start:right + 1] in forbidden_set:
                    # If we find a forbidden substring, move left pointer
                    # to just after the start of this forbidden substring
                    left = start + 1
                    break
            
            # Update the maximum length
            max_length = max(max_length, right - left + 1)
        
        return max_length