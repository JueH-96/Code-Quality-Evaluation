class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        # Count of '1's in the original string
        c1 = s.count('1')
        # Build the augmented string t = '1' + s + '1'
        t = '1' + s + '1'
        # Parse t into blocks of consecutive '0's or '1's
        blocks = []
        cur_char = t[0]
        cur_count = 1
        for ch in t[1:]:
            if ch == cur_char:
                cur_count += 1
            else:
                blocks.append((cur_char, cur_count))
                cur_char = ch
                cur_count = 1
        blocks.append((cur_char, cur_count))
        
        M = len(blocks)
        # Compute the largest zero‐block in the original t
        original_max_zeros = 0
        for char, cnt in blocks:
            if char == '0' and cnt > original_max_zeros:
                original_max_zeros = cnt
        
        max_net_gain = 0
        # Try removing each ones‐block that is surrounded by zeros
        # A ones‐block is at index i if blocks[i][0]=='1', and i>0, i<M-1,
        # and its neighbors blocks[i-1] and blocks[i+1] are zeros‐blocks.
        for i in range(1, M-1):
            char, x = blocks[i]
            if char == '1':
                # Check both sides are zeros‐blocks
                left_char, Z_left = blocks[i-1]
                right_char, Z_right = blocks[i+1]
                if left_char == '0' and right_char == '0':
                    # If we remove this ones‐block of size x, we lose x ones.
                    # After removal we can fill a zero‐block.  The best zero‐block
                    # to fill is either the newly fused block (Z_left + x + Z_right)
                    # or some original zeros‐block elsewhere (size original_max_zeros).
                    # Net gain = new_ones_filled - ones_lost.
                    #   fused option:   (Z_left + x + Z_right) - x = Z_left + Z_right
                    #   far‐away option: original_max_zeros - x
                    fused_gain = Z_left + Z_right
                    far_gain = original_max_zeros - x
                    net_gain = fused_gain if fused_gain > far_gain else far_gain
                    if net_gain > max_net_gain:
                        max_net_gain = net_gain
        
        # If no valid removal, max_net_gain stays 0 -> return c1
        # Otherwise, return c1 + best net gain
        return c1 + max_net_gain