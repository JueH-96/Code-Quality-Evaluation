class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        # Augment the string with '1' at both ends
        t = '1' + s + '1'
        
        # Find all blocks of '0's surrounded by '1's
        zero_blocks = []
        i = 0
        while i < len(t):
            if t[i] == '0':
                start = i
                while i < len(t) and t[i] == '0':
                    i += 1
                end = i
                zero_blocks.append((start, end))
            else:
                i += 1
        
        # Calculate the number of '1's in the original string
        original_active_count = s.count('1')
        
        # If no zero blocks, return the original count
        if not zero_blocks:
            return original_active_count
        
        # Try trading each zero block and calculate the maximum active sections
        max_active = original_active_count
        for start, end in zero_blocks:
            # Calculate the number of '1's we can gain by converting this zero block
            gain = end - start
            # Calculate the number of '1's we lose by converting the surrounding '1's to '0's
            lose = 0
            if start > 0 and t[start - 1] == '1':
                lose += 1
            if end < len(t) and t[end] == '1':
                lose += 1
            # Calculate the new active count after this trade
            new_active_count = original_active_count + gain - lose
            # Update the maximum active sections
            max_active = max(max_active, new_active_count)
        
        return max_active