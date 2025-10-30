class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        # Augment string with 1's at both ends
        s = '1' + s + '1'
        n = len(s)
        
        # If no trade is possible, count original active sections
        def count_ones(s):
            return sum(1 for c in s[1:-1] if c == '1')
        
        # Find all blocks of 1's surrounded by 0's
        def find_one_blocks():
            blocks = []
            start = -1
            for i in range(1, n-1):
                if s[i] == '1' and s[i-1] == '0':
                    start = i
                elif s[i] == '0' and start != -1:
                    if s[i-1] == '1':
                        blocks.append((start, i-1))
                    start = -1
            if start != -1 and s[n-2] == '1':
                blocks.append((start, n-2))
            return blocks
        
        # Find all blocks of 0's surrounded by 1's
        def find_zero_blocks():
            blocks = []
            start = -1
            for i in range(1, n-1):
                if s[i] == '0' and s[i-1] == '1':
                    start = i
                elif s[i] == '1' and start != -1:
                    blocks.append((start, i-1))
                    start = -1
            if start != -1 and s[n-2] == '0':
                if s[n-1] == '1':
                    blocks.append((start, n-2))
            return blocks
        
        # Get original count
        max_active = count_ones(s)
        
        # Try all possible trades
        one_blocks = find_one_blocks()
        zero_blocks = find_zero_blocks()
        
        # For each possible trade, simulate and count
        for one_block in one_blocks:
            one_len = one_block[1] - one_block[0] + 1
            for zero_block in zero_blocks:
                zero_len = zero_block[1] - zero_block[0] + 1
                # Calculate new active sections after trade
                new_active = count_ones(s) - one_len + zero_len
                max_active = max(max_active, new_active)
        
        return max_active