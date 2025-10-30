from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # Use DP tracking best score ending in each parity (0=even,1=odd)
        NEG_INF = -10**30
        best = [NEG_INF, NEG_INF]
        
        # Initialize with nums[0]
        p0 = nums[0] & 1
        best[p0] = nums[0]
        
        # Process the rest of the array
        for num in nums[1:]:
            p = num & 1
            # If we come from same parity, no penalty
            take_same = best[p] + num
            # If we come from different parity, apply penalty x
            take_diff = best[1-p] - x + num
            # Best possible taking this num
            cand = take_same if take_same > take_diff else take_diff
            # Update the best for this parity
            if cand > best[p]:
                best[p] = cand
        
        # The answer is the best over both parities
        return max(best[0], best[1])