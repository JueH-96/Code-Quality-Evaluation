class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        # Augment the string
        t = '1' + s + '1'
        n = len(t)
        
        # Find all blocks of '1's surrounded by '0's
        one_blocks = []
        i = 0
        while i < n:
            if t[i] == '1':
                j = i
                while j < n and t[j] == '1':
                    j += 1
                if i > 0 and j < n and t[i-1] == '0' and t[j] == '0':
                    one_blocks.append((i, j))
                i = j
            else:
                i += 1
        
        # Count current '1's
        current_ones = s.count('1')
        max_ones = current_ones
        
        # Try each possible trade
        for start1, end1 in one_blocks:
            # Apply first part of trade
            t_list = list(t)
            for k in range(start1, end1):
                t_list[k] = '0'
            t_copy = ''.join(t_list)
            
            # Find zero blocks in modified string
            i = 0
            while i < n:
                if t_copy[i] == '0':
                    j = i
                    while j < n and t_copy[j] == '0':
                        j += 1
                    if i > 0 and j < n and t_copy[i-1] == '1' and t_copy[j] == '1':
                        # Apply second part of trade
                        t_list2 = list(t_copy)
                        for k in range(i, j):
                            t_list2[k] = '1'
                        
                        # Count '1's in original positions
                        count = sum(1 for idx in range(1, len(s) + 1) if t_list2[idx] == '1')
                        max_ones = max(max_ones, count)
                    
                    i = j
                else:
                    i += 1
        
        return max_ones