class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        """
        Sliding–window approach.
        Keep a window [left, right] such that every character appears
        at most twice.  Extend `right` one step at a time; if adding the
        new character breaks the rule, shrink the window from the left
        until the rule is satisfied again.  Track the largest window
        seen.
        """
        from collections import defaultdict
        
        counts = defaultdict(int)          # frequency of chars in window
        left = 0
        best = 0
        
        for right, ch in enumerate(s):
            counts[ch] += 1                       # include s[right]
            
            # shrink window while current char occurs more than twice
            while counts[ch] > 2:
                counts[s[left]] -= 1
                left += 1
            
            # update answer
            best = max(best, right - left + 1)
        
        return best