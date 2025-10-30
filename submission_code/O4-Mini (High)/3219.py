from typing import List

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        if n <= 1:
            return nums
        
        # Compute connected components on the value graph (threshold graph)
        unique_vals = sorted(set(nums))
        comp_id = {}
        comp = 0
        comp_end = unique_vals[0] + limit
        comp_id[unique_vals[0]] = comp
        for v in unique_vals[1:]:
            if v <= comp_end:
                # Still in the same component
                comp_id[v] = comp
                comp_end = max(comp_end, v + limit)
            else:
                # Start a new component
                comp += 1
                comp_id[v] = comp
                comp_end = v + limit
        
        # Gather all values by their component
        comps_count = comp + 1
        values_by_comp = [[] for _ in range(comps_count)]
        for v in nums:
            values_by_comp[comp_id[v]].append(v)
        
        # Sort each component's values so we can assign smallest first
        for lst in values_by_comp:
            lst.sort()
        
        # Build the result by assigning, at each index i, the smallest remaining
        # value from the component of nums[i].
        res = [0] * n
        ptr = [0] * comps_count
        for i, v in enumerate(nums):
            cid = comp_id[v]
            res[i] = values_by_comp[cid][ptr[cid]]
            ptr[cid] += 1
        
        return res