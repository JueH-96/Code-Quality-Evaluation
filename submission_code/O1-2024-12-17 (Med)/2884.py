class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # Convert the list of forbidden strings to a set for O(1) membership checks
        forbidden_set = set(forbidden)
        
        left = 0
        max_length = 0
        n = len(word)
        
        for right in range(n):
            # Only need to check up to 10 characters back (max length of forbidden strings)
            start_check = max(left, right - 9)
            # Check all substrings ending at 'right' and starting from indices in [start_check..right]
            for start_idx in range(right, start_check - 1, -1):
                substr = word[start_idx:right + 1]
                if substr in forbidden_set:
                    # If this substring is forbidden, move 'left'
                    left = start_idx + 1
                    # No need to check smaller substrings if we've already invalidated this region
                    break
            
            # Update the maximum valid substring length
            max_length = max(max_length, right - left + 1)
        
        return max_length