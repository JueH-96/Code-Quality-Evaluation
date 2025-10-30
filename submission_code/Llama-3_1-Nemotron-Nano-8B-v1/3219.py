from typing import List

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        # Create a list of tuples (value, original index)
        sorted_nums = sorted((num, i) for i, num in enumerate(nums))
        n = len(sorted_nums)
        if n == 0:
            return []
        
        # Split into groups where consecutive elements have difference <= limit
        groups = []
        current_group = []
        for num, idx in sorted_nums:
            if not current_group:
                current_group.append((num, idx))
            else:
                if num - current_group[-1][0] <= limit:
                    current_group.append((num, idx))
                else:
                    groups.append(current_group)
                    current_group = [(num, idx)]
        if current_group:
            groups.append(current_group)
        
        # Prepare the result array
        res = [0] * n
        for group in groups:
            # Extract and sort the original indices
            indices = sorted(idx for num, idx in group)
            # Extract the values (already sorted)
            values = [num for num, idx in group]
            # Assign the values to the sorted indices
            for i, idx in enumerate(indices):
                res[idx] = values[i]
        
        return res