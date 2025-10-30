class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        n = len(s)
        original_ones = s.count('1')
        
        # The trade can only be beneficial if we can find a 1-block surrounded by 0s and a 0-block surrounded by 1s
        # We need to find all possible 1-blocks surrounded by 0s (in the augmented string)
        # and all 0-blocks surrounded by 1s (in the augmented string)
        
        # The augmented string is '1' + s + '1', but we don't need to construct it explicitly
        
        # First, find all 1-blocks that are surrounded by 0s in the original string (or augmented)
        one_blocks = []
        i = 0
        while i < n:
            if s[i] == '1':
                start = i
                while i < n and s[i] == '1':
                    i += 1
                end = i - 1
                # Check if this block is surrounded by 0s (or boundaries, which are treated as 1s)
                left_is_zero = (start > 0 and s[start - 1] == '0') or (start == 0)
                right_is_zero = (end < n - 1 and s[end + 1] == '0') or (end == n - 1)
                # In augmented string, left boundary is 1, right boundary is 1
                # So the 1-block is surrounded by 0s in original string only if:
                # (start >0 and s[start-1] is 0) AND (end <n-1 and s[end+1] is 0)
                # Or start ==0 and end <n-1 and s[end+1] is 0 (augmented left is 1, but original left is start 0)
                # Or end ==n-1 and start >0 and s[start-1] is 0
                # Or start ==0 and end ==n-1 (entire string is 1s, but augmented is 1s around, so no 0s around)
                # So the block is surrounded by 0s in augmented string only if in original string:
                # (start >0 and s[start-1] == '0') and (end <n-1 and s[end+1] == '0')
                # Because augmented string adds 1s at boundaries, so any 1-block touching the boundary is not surrounded by 0s in augmented string.
                if (start > 0 and s[start - 1] == '0') and (end < n - 1 and s[end + 1] == '0'):
                    one_blocks.append((start, end))
            else:
                i += 1
        
        # Now find all 0-blocks that are surrounded by 1s (including augmented boundaries)
        zero_blocks = []
        i = 0
        while i < n:
            if s[i] == '0':
                start = i
                while i < n and s[i] == '0':
                    i += 1
                end = i - 1
                # Check if surrounded by 1s in augmented string
                # left is 1 if start ==0 (augmented) or s[start-1] == '1'
                left_ok = (start == 0) or (s[start - 1] == '1')
                # right is 1 if end ==n-1 (augmented) or s[end+1] == '1'
                right_ok = (end == n - 1) or (s[end + 1] == '1')
                if left_ok and right_ok:
                    zero_blocks.append((start, end))
            else:
                i += 1
        
        if not one_blocks or not zero_blocks:
            return original_ones
        
        max_gain = 0
        
        # The best possible gain is (zero_block_length) - (one_block_length)
        # We want to find the maximum (zero_block_length - one_block_length)
        # So the optimal is to choose the smallest one_block and largest zero_block
        
        min_one_len = min(end - start + 1 for start, end in one_blocks)
        max_zero_len = max(end - start + 1 for start, end in zero_blocks)
        max_gain = max_zero_len - min_one_len
        
        if max_gain > 0:
            return original_ones + max_gain
        else:
            return original_ones