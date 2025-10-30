class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        left = 0
        count = 0  # count of adjacent duplicate pairs in the current window
        max_len = 1
        
        for right in range(n):
            # If we have a duplicate pair at the current right index
            if right > left and s[right] == s[right - 1]:
                count += 1
            
            # If there are more than one pair in the window, shrink from the left.
            while count > 1:
                # If the pair we are about to remove (starting at left) contributed to count, reduce count.
                if left + 1 < n and s[left] == s[left + 1]:
                    count -= 1
                left += 1
            
            # Update the maximum valid window length.
            max_len = max(max_len, right - left + 1)
        
        return max_len