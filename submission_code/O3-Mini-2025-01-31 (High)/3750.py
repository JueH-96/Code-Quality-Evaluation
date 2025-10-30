from typing import List
from collections import defaultdict

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        # Build dictionary: value -> list of indices (positions are naturally in increasing order)
        positions = defaultdict(list)
        for i, v in enumerate(nums):
            positions[v].append(i)
        
        # Precompute the answer for each index in nums.
        # For an index i, the answer is the minimum circular distance to any other index j
        # that has the same value.
        # If the value appears only once, answer[i] will be -1.
        ans_by_index = [-1] * n
        
        for indices in positions.values():
            # If this value appears only once, then there's no other index.
            if len(indices) == 1:
                ans_by_index[indices[0]] = -1
            else:
                m = len(indices)
                # For each occurrence, consider its two neighbors in the circular list.
                for j in range(m):
                    curr = indices[j]
                    # Left neighbor (with circular wrap-around)
                    left = indices[j - 1]  # When j==0, Python negative index gives the last element.
                    # Right neighbor (with circular wrap-around)
                    right = indices[j + 1] if j < m - 1 else indices[0]
                    
                    # Compute circular distance: For two indices a and b in a circular array,
                    # distance is min(|a - b|, n - |a - b|).
                    diff_left = abs(curr - left)
                    diff_right = abs(curr - right)
                    d1 = min(diff_left, n - diff_left)
                    d2 = min(diff_right, n - diff_right)
                    
                    ans_by_index[curr] = min(d1, d2)
                    
        # Answer each query using the precomputed values.
        return [ans_by_index[q] for q in queries]