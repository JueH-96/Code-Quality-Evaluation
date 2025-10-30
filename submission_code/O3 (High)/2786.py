class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        # Sliding–window approach.
        # `left`  – left end of current window
        # `pairs` – number of equal-adjacent pairs inside the window
        left = 0
        pairs = 0
        best = 1                                   # string length ≥ 1
        
        for right in range(len(s)):
            # When the new character forms an equal–adjacent pair with the
            # previous one, we add that pair to the counter.
            if right > 0 and s[right] == s[right - 1]:
                pairs += 1
            
            # Shrink the window from the left until we have at most one pair.
            while pairs > 1:
                if s[left] == s[left + 1]:
                    pairs -= 1                     # the pair (left,left+1) leaves window
                left += 1
            
            # Update the maximum window length found so far.
            best = max(best, right - left + 1)
        
        return best