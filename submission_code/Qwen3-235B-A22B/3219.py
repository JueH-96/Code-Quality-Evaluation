from typing import List

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        sorted_vals = sorted((val, idx) for idx, val in enumerate(nums))
        groups = []
        current_group = []
        prev_val = None
        
        for val, idx in sorted_vals:
            if not current_group:
                current_group.append((val, idx))
            else:
                if val - prev_val <= limit:
                    current_group.append((val, idx))
                else:
                    groups.append(current_group)
                    current_group = [(val, idx)]
            prev_val = val
        
        if current_group:
            groups.append(current_group)
        
        res = [0] * len(nums)
        for group in groups:
            values = [val for val, idx in group]
            positions = [idx for val, idx in group]
            positions.sort()
            for i in range(len(values)):
                res[positions[i]] = values[i]
        
        return res