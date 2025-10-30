from typing import List
from bisect import bisect_left

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        # Build a map from value to sorted list of indices where it appears
        positions = {}
        for idx, val in enumerate(nums):
            positions.setdefault(val, []).append(idx)
        for val in positions:
            positions[val].sort()
        
        ans = []
        for q in queries:
            val = nums[q]
            idx_list = positions[val]
            if len(idx_list) < 2:
                # No other occurrence
                ans.append(-1)
                continue
            # Find insertion point of q in the sorted list
            pos = bisect_left(idx_list, q)
            # Neighbors in the sorted list (circularly)
            # next occurrence (wrap around)
            nxt = idx_list[(pos + 1) % len(idx_list)]
            # previous occurrence (wrap around)
            prv = idx_list[(pos - 1) % len(idx_list)]
            # Compute circular distances
            d1 = abs(nxt - q)
            d1 = min(d1, n - d1)
            d2 = abs(q - prv)
            d2 = min(d2, n - d2)
            ans.append(min(d1, d2))
        return ans

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.solveQueries([1,3,1,4,1,3,2], [0,3,5]))  # [2, -1, 3]
    print(sol.solveQueries([1,2,3,4], [0,1,2,3]))      # [-1, -1, -1, -1]