from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        ALL_BITS = (1 << 30) - 1
        n = len(nums)
        
        # -------------------------------------------------------------
        # helper : can we reach a result whose OR is contained in `mask`
        #          (i.e. every segment's AND is a subset of mask)
        #          using at most k merges ?
        # -------------------------------------------------------------
        def feasible(mask: int) -> bool:
            segments = 0
            cur_and = ALL_BITS          # AND of current (not yet closed) segment
            
            for x in nums:
                cur_and &= x
                # if current segment is already "good", close it
                if (cur_and & ~mask) == 0:
                    segments += 1
                    cur_and = ALL_BITS   # start new segment
            
            # no segment has been closed => whole array is one segment
            if segments == 0:
                return (cur_and & ~mask) == 0 and n - 1 <= k    # need n-1 merges
            
            # minimum merges necessary = n - segments
            return n - segments <= k
        
        # -------------------------------------------------------------
        # Start from everything allowed, erase (high) bits if possible
        # -------------------------------------------------------------
        ans = ALL_BITS
        for b in reversed(range(30)):              # from bit 29 down to 0
            candidate = ans & ~(1 << b)            # try to clear this bit
            if feasible(candidate):                # if still achievable => keep it 0
                ans = candidate
        
        return ans