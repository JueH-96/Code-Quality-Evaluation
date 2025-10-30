from typing import List
import math
import sys

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        # Precompute the smallest index for each element value.
        # An element value v is eligible for a group if group % v == 0.
        # Since if there are several eligible elements, we want the one with the lowest index,
        # we only need to know the minimal index for each value.
        #
        # We then precompute for every number m (up to max value in groups or elements)
        # the minimum index among all candidates v (from elements) that divide m.
        # This is done by iterating over candidate values and updating all multiples.
        
        if not groups or not elements:
            return [-1] * len(groups)
        
        # max_val is the maximum value in both groups and elements.
        max_val = max(max(groups), max(elements))
        INF = float('inf')
        
        # candidate[v] stores the smallest index of an element with value v.
        candidate = [INF] * (max_val + 1)
        for j, val in enumerate(elements):
            # Update candidate for the value if this index is smaller.
            if val <= max_val:
                candidate[val] = min(candidate[val], j)
                
        # best[m] stores the minimum candidate index among all element values v that divide m.
        best = [INF] * (max_val + 1)
        # For every possible value v that appears in elements (i.e. candidate[v] != INF),
        # update all multiples of v.
        for v in range(1, max_val + 1):
            if candidate[v] != INF:
                # Iterate through all multiples of v up to max_val.
                # For each m, if candidate[v] is smaller than the current best[m], update it.
                for m in range(v, max_val + 1, v):
                    if candidate[v] < best[m]:
                        best[m] = candidate[v]
        
        # Build the answer for each group.
        # For a group value, if best[group] is not INF, then that index is the smallest eligible element index.
        # Otherwise, assign -1.
        ans = []
        for group in groups:
            ans.append(best[group] if best[group] != INF else -1)
        return ans

# For local testing purposes:
if __name__ == '__main__':
    sol = Solution()
    # Example 1
    groups1 = [8, 4, 3, 2, 4]
    elements1 = [4, 2]
    print(sol.assignElements(groups1, elements1))  # Expected output: [0, 0, -1, 1, 0]

    # Example 2
    groups2 = [2, 3, 5, 7]
    elements2 = [5, 3, 3]
    print(sol.assignElements(groups2, elements2))  # Expected output: [-1, 1, 0, -1]

    # Example 3
    groups3 = [10, 21, 30, 41]
    elements3 = [2, 1]
    print(sol.assignElements(groups3, elements3))  # Expected output: [0, 1, 0, 1]