class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # Convert list of forbidden substrings into a set for faster membership checks
        forbidden_set = set(forbidden)
        
        max_len = 0
        start = 0
        n = len(word)
        
        # We'll keep expanding a window [start..right].
        # For each right, check up to the last 10 characters for any forbidden substring.
        for right in range(n):
            # We'll track the farthest we have to move 'start' to keep things valid
            new_start = start  
            
            # We only need to look back at most 10 characters
            for length in range(1, 11):
                if (right - length + 1) < start:
                    break  # No need to check if we've already passed beyond the current start
                candidate = word[right - length + 1 : right + 1]
                if candidate in forbidden_set:
                    # If "candidate" is forbidden, we must ensure our valid substring
                    # starts after this forbidden region
                    new_start = max(new_start, right - length + 2)
            
            # Update start to the largest shift needed
            start = new_start
            # Update the max length
            max_len = max(max_len, right - start + 1)
        
        return max_len