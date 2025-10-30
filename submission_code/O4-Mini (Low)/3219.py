from typing import List

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        # Pair each number with its index and sort by number
        arr = sorted((val, i) for i, val in enumerate(nums))
        
        res = [0] * n
        # We'll identify connected components in the "value graph"
        # where edges connect values within 'limit' difference.
        comp_vals = [arr[0][0]]
        comp_idxs = [arr[0][1]]
        
        for k in range(1, n):
            val, idx = arr[k]
            prev_val, _ = arr[k-1]
            # If difference <= limit, stay in same component
            if val - prev_val <= limit:
                comp_vals.append(val)
                comp_idxs.append(idx)
            else:
                # Process the previous component
                comp_vals.sort()
                comp_idxs.sort()
                for vi, pi in zip(comp_vals, comp_idxs):
                    res[pi] = vi
                # Start a new component
                comp_vals = [val]
                comp_idxs = [idx]
        
        # Don't forget the last component
        comp_vals.sort()
        comp_idxs.sort()
        for vi, pi in zip(comp_vals, comp_idxs):
            res[pi] = vi
        
        return res