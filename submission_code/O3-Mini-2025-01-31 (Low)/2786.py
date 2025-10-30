class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # sliding window pointers
        l = 0
        pairs_count = 0  # count of consecutive duplicates in current window
        best = 1
        
        for r in range(n):
            # When extending the window: if current char equals previous char,
            # then we have a new consecutive duplicate pair.
            if r > l and s[r] == s[r - 1]:
                pairs_count += 1
            
            # If more than allowed number of consecutive duplicate pairs, shrink window from left
            while pairs_count > 1:
                # If s[l] and s[l+1] formed a consecutive duplicate pair, remove that pair count.
                if l + 1 < n and s[l] == s[l + 1]:
                    pairs_count -= 1
                l += 1  # Move left pointer
            # update best window length (always valid window)
            best = max(best, r - l + 1)
        return best