from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # dp_prefix[i][j]: set of possible OR-sums using j elements from nums[0...i-1]
        dp_prefix = [[set() for _ in range(k + 1)] for _ in range(n + 1)]
        dp_prefix[0][0] = {0}
        
        for i in range(1, n + 1):
            num = nums[i - 1]
            for j in range(k + 1):
                # Case 1: Don't include nums[i-1].
                dp_prefix[i][j].update(dp_prefix[i - 1][j])
                
                # Case 2: Include nums[i-1].
                if j > 0:
                    for val in dp_prefix[i - 1][j - 1]:
                        dp_prefix[i][j].add(val | num)

        # dp_suffix[i][j]: set of possible OR-sums using j elements from nums[i...n-1]
        dp_suffix = [[set() for _ in range(k + 1)] for _ in range(n + 1)]
        dp_suffix[n][0] = {0}
        
        for i in range(n - 1, -1, -1):
            num = nums[i]
            for j in range(k + 1):
                # Case 1: Don't include nums[i].
                dp_suffix[i][j].update(dp_suffix[i + 1][j])
                
                # Case 2: Include nums[i].
                if j > 0:
                    for val in dp_suffix[i + 1][j - 1]:
                        dp_suffix[i][j].add(val | num)
                        
        max_val = 0
        
        # A split point `s` means the first k items are from nums[0...s]
        # and the second k items are from nums[s+1...n-1].
        # Length of prefix `s+1` must be at least `k`, so `s >= k-1`.
        # Length of suffix `n-(s+1)` must be at least `k`, so `s <= n-k-1`.
        for s in range(k - 1, n - k):
            prefix_sums = dp_prefix[s + 1][k]
            suffix_sums = dp_suffix[s + 1][k]
            
            if not prefix_sums or not suffix_sums:
                continue

            # To find max(A XOR B) for A in prefix_sums, B in suffix_sums, use a Trie.
            # Numbers are < 2^7, so 7 bits are sufficient.
            trie = {}
            for s_val in suffix_sums:
                node = trie
                for bit_pos in range(6, -1, -1):
                    bit = (s_val >> bit_pos) & 1
                    if bit not in node:
                        node[bit] = {}
                    node = node[bit]

            current_max_for_split = 0
            for p_val in prefix_sums:
                node = trie
                current_xor = 0
                for bit_pos in range(6, -1, -1):
                    p_bit = (p_val >> bit_pos) & 1
                    # To maximize XOR, we want a bit that is opposite to p_bit.
                    target_bit = 1 - p_bit
                    if target_bit in node:
                        # If found, this bit contributes 1 to the XOR result.
                        current_xor |= (1 << bit_pos)
                        node = node[target_bit]
                    else:
                        # Otherwise, we must take the path with p_bit.
                        node = node[p_bit]
                current_max_for_split = max(current_max_for_split, current_xor)
            
            max_val = max(max_val, current_max_for_split)
            
        return max_val