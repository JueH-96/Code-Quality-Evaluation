class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        # l: left pointer of current window
        # count: number of consecutive equal-digit pairs in the window
        l = 0
        count = 0
        # At least one character exists, so min answer is 1
        ans = 1
        
        for r in range(n):
            # If the newly added character at r forms a pair with r-1, increment count
            if r > 0 and s[r] == s[r - 1]:
                count += 1
            
            # If we have more than one pair, move the left pointer to restore validity
            while count > 1:
                # If removing the character at l drops a counted pair at (l, l+1), decrement
                if s[l] == s[l + 1]:
                    count -= 1
                l += 1
            
            # Update the maximum valid window length
            ans = max(ans, r - l + 1)
        
        return ans