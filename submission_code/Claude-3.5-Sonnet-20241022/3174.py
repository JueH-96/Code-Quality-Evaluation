class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        total_changes = 0
        
        # Check each pair of adjacent characters
        for i in range(0, n, 2):
            # Count changes needed to make the pair either "00" or "11"
            zeros = (s[i] == '1') + (s[i+1] == '1')  # Changes needed for "00"
            ones = (s[i] == '0') + (s[i+1] == '0')   # Changes needed for "11"
            
            # Take minimum of changes needed
            total_changes += min(zeros, ones)
            
        return total_changes