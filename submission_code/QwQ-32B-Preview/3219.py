from typing import List

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        # Step 1: Sort the array with original indices
        sorted_with_indices = sorted([(value, index) for index, value in enumerate(nums)])
        
        # Step 2: Group the sorted elements based on difference <= limit
        groups = []
        current_group = []
        for i in range(len(sorted_with_indices)):
            if i == 0 or sorted_with_indices[i][0] - sorted_with_indices[i-1][0] <= limit:
                current_group.append(sorted_with_indices[i])
            else:
                groups.append(current_group)
                current_group = [sorted_with_indices[i]]
        # Don't forget to add the last group
        if current_group:
            groups.append(current_group)
        
        # Step 3: For each group, assign sorted values to sorted indices
        result = [0] * len(nums)
        for group in groups:
            values = [pair[0] for pair in group]
            indices = [pair[1] for pair in group]
            values.sort()
            indices.sort()
            for value, index in zip(values, indices):
                result[index] = value
        return result