from typing import List

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        n = len(elements)
        INF = n  # A value greater than any valid index (indices: 0 .. n-1)
        # Determine the maximum value we need to consider.
        # We need it for the sieve range so that we cover all possible group/elements values.
        max_group = max(groups) if groups else 0
        max_elem_val = max(elements) if elements else 0
        M = max(max_group, max_elem_val)
        
        # candidate[x] will store the smallest index j such that elements[j] equals x.
        candidate = [INF] * (M + 1)
        for j, val in enumerate(elements):
            # Update candidate for this element value.
            # An element value might appear multiple times, and we need the smallest index.
            if val <= M:
                candidate[val] = min(candidate[val], j)
        
        # best[x] will store the smallest candidate index among all element values that divide x.
        # In other words, best[x] = min { candidate[v] such that v divides x }.
        best = [INF] * (M + 1)
        for v in range(1, M + 1):
            if candidate[v] != INF:
                # For every multiple m of v, update best[m]
                for m in range(v, M + 1, v):
                    if candidate[v] < best[m]:
                        best[m] = candidate[v]
        
        # Build the result: for each group in 'groups', if best[group] is updated (i.e. < INF),
        # then assign that element's index; otherwise assign -1.
        return [best[g] if best[g] != INF else -1 for g in groups]