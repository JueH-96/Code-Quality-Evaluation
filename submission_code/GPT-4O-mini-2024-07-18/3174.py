class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        count0 = count1 = 0
        
        # Count the number of '0's and '1's in even and odd positions
        for i in range(0, n, 2):
            if s[i] == '0':
                count0 += 1
            else:
                count1 += 1
        
        count0_even = count0
        count1_even = count1
        
        count0 = count1 = 0
        
        for i in range(1, n, 2):
            if s[i] == '0':
                count0 += 1
            else:
                count1 += 1
        
        count0_odd = count0
        count1_odd = count1
        
        # Calculate the minimum changes needed for both patterns
        changes_pattern1 = (count1_even + count0_odd)  # Make even positions '0' and odd positions '1'
        changes_pattern2 = (count0_even + count1_odd)  # Make even positions '1' and odd positions '0'
        
        return min(changes_pattern1, changes_pattern2)