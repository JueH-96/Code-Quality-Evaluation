from typing import List
from collections import defaultdict

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        # Map each value to the list of its indices
        pos = defaultdict(list)
        for i, v in enumerate(nums):
            pos[v].append(i)
        
        # Prepare answer array initialized to -1
        ans = [-1] * n
        
        # For each group of same-value indices, compute minimal circular distance
        for v, idx_list in pos.items():
            m = len(idx_list)
            if m <= 1:
                # Only one occurrence => answers stay -1
                continue
            idx_list.sort()
            for k in range(m):
                curr = idx_list[k]
                prev = idx_list[(k - 1) % m]
                nxt  = idx_list[(k + 1) % m]
                # circular distances
                dprev = (curr - prev + n) % n
                dnext = (nxt - curr + n) % n
                ans[curr] = min(dprev, dnext)
        
        # Build the result for the queries
        return [ans[i] for i in queries]


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.solveQueries([1,3,1,4,1,3,2], [0,3,5]))  # [2, -1, 3]
    print(sol.solveQueries([1,2,3,4], [0,1,2,3]))      # [-1, -1, -1, -1]