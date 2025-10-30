class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        if n % 2 != 0:
            return -1
        
        min_changes = float('inf')
        
        # Try to make the entire string 0's or 1's
        count_0 = s.count('0')
        count_1 = n - count_0
        min_changes = min(min_changes, count_0, count_1)
        
        # Try to partition the string into substrings of even length
        for i in range(1, n, 2):
            if s[i] != s[0]:
                min_changes = min(min_changes, 1)
            if s[i-1] != s[0]:
                min_changes = min(min_changes, 1)
        
        return min_changes