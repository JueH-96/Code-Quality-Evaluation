class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        # Parse the string into blocks of consecutive '0's and '1's
        blocks = []
        prev_char = s[0]
        count = 1
        for c in s[1:]:
            if c == prev_char:
                count += 1
            else:
                blocks.append((prev_char, count))
                prev_char = c
                count = 1
        blocks.append((prev_char, count))
        
        # Calculate the original number of active sections
        original_ones = sum(cnt for ch, cnt in blocks if ch == '1')
        max_gain = 0
        
        # Check each block to find valid trades
        for i in range(len(blocks)):
            ch, cnt = blocks[i]
            if ch == '1':
                if i > 0 and i < len(blocks) - 1:
                    if blocks[i-1][0] == '0' and blocks[i+1][0] == '0':
                        current_gain = blocks[i-1][1] + blocks[i+1][1]
                        if current_gain > max_gain:
                            max_gain = current_gain
        
        return original_ones + max_gain