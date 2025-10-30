from typing import List
from collections import defaultdict
from bisect import bisect_left
from functools import lru_cache

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        # We reinterpret the problem as: we are allowed up to (k+1) blocks
        # (each block is a maximal group of identical numbers in the subsequence).
        # We want to pick a subsequence from nums that is "good" (at most k transitions),
        # meaning at most k+1 blocks.
        n = len(nums)
        maxBlocks = k + 1
        
        # Precompute positions for each distinct value
        positions = defaultdict(list)
        for i, v in enumerate(nums):
            positions[v].append(i)
        
        # Use DP with memoization.
        # dp(i, b): maximum length of a good subsequence that can be achieved
        # starting from index i using at most b blocks.
        @lru_cache(maxsize=None)
        def dp(i: int, b: int) -> int:
            if i >= n or b == 0:
                return 0
            best = 0
            # Iterate over each distinct value v that appears in the array
            for v, p_list in positions.items():
                # Find the first occurrence of v with index >= i.
                pos = bisect_left(p_list, i)
                if pos == len(p_list):
                    continue  # v does not appear at or after index i.
                # p_list[pos:] are valid indices for v.
                # We want to choose a block consisting of some (non-empty) subsequence of these indices.
                # The block must be in order, and we can choose how many occurrences to take.
                current_count = 0
                # We'll iterate over the candidate indices to be the "last" index in this block.
                for j in p_list[pos:]:
                    current_count += 1
                    # current_count occurrences of v are picked and the block ends at index j.
                    candidate = current_count + dp(j + 1, b - 1)
                    if candidate > best:
                        best = candidate
            return best
        
        return dp(0, maxBlocks)