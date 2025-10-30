class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        # Count original active sections (ones in s)
        original_ones = s.count('1')
        n = len(s)
        # Build augmented string t = '1' + s + '1'
        t = '1' + s + '1'
        # Parse t into blocks of consecutive chars
        zero_blocks = []  # lengths of zero runs
        one_blocks = []   # lengths of one runs
        prev = t[0]
        cnt = 1
        for c in t[1:]:
            if c == prev:
                cnt += 1
            else:
                # flush the previous block
                if prev == '0':
                    zero_blocks.append(cnt)
                else:
                    one_blocks.append(cnt)
                prev = c
                cnt = 1
        # flush last
        if prev == '0':
            zero_blocks.append(cnt)
        else:
            one_blocks.append(cnt)
        m = len(zero_blocks)
        # If there are fewer than 2 zero-blocks, there are fewer than 3 one-blocks
        # so no interior one-block can be removed. Return original count.
        if m < 2:
            return original_ones
        
        # Precompute prefix max and suffix max of zero_blocks
        # prefix_max[k] = max(zero_blocks[0..k-1]) for k in [0..m]
        prefix_max = [0] * (m + 1)
        for k in range(1, m + 1):
            prefix_max[k] = max(prefix_max[k-1], zero_blocks[k-1])
        # suffix_max[k] = max(zero_blocks[k..m-1]) for k in [0..m], with suffix_max[m]=0
        suffix_max = [0] * (m + 1)
        for k in range(m-1, -1, -1):
            suffix_max[k] = max(zero_blocks[k], suffix_max[k+1])
        
        ans = original_ones
        # Interior one-blocks are one_blocks[1..m-1]
        # zero_blocks indices j=0..m-1 correspond to gaps between one_blocks
        # Removing one_blocks[i] (i in 1..m-1) merges zero_blocks[i-1] and zero_blocks[i]
        for i in range(1, m):
            L1 = one_blocks[i]
            j = i - 1
            # merged zero block length
            Z_new = zero_blocks[j] + L1 + zero_blocks[j+1]
            # best zero-block to flip is max of Z_new and any other original zero block
            # other zero-block max excludes indices j and j+1
            other_max = max(prefix_max[j], suffix_max[j+2] if j+2 <= m else 0)
            best_zero = max(Z_new, other_max)
            # net ones = original_ones - removed ones + flipped zeros
            cand = original_ones - L1 + best_zero
            if cand > ans:
                ans = cand
        
        return ans