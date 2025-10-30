from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        # Map each value to its list of indices
        pos_map = {}
        for i, v in enumerate(nums):
            pos_map.setdefault(v, []).append(i)
        
        ans = float('inf')
        # For each value, compute the maximal minimal distance to nearest occurrence
        for positions in pos_map.values():
            m = len(positions)
            # If every element is this value, zero seconds needed
            if m == n:
                return 0
            max_gap_half = 0
            # Compute gaps between consecutive positions
            for i in range(m - 1):
                d = positions[i+1] - positions[i]
                max_gap_half = max(max_gap_half, d // 2)
            # Circular gap: from last to first via wrap
            circular_gap = (positions[0] + n) - positions[-1]
            max_gap_half = max(max_gap_half, circular_gap // 2)
            # Keep the minimum over all values
            ans = min(ans, max_gap_half)
        
        return ans