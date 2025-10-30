from typing import List
from collections import defaultdict

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Function to count the number of set bits in a number
        def bit_count(x: int) -> int:
            return bin(x).count('1')
        
        # Dictionary mapping bit-count -> list of (index, value)
        groups = defaultdict(list)
        
        # Fill the groups with the indices and values
        for i, val in enumerate(nums):
            groups[bit_count(val)].append((i, val))
        
        # For each bit-count group, sort by the values
        # and place them back into the positions in ascending order
        new_arr = [0] * len(nums)
        for bcount, arr in groups.items():
            # Sort indices by ascending index, sort values by ascending value
            arr.sort(key=lambda x: x[1])  # sort by the value
            # Extract indices in ascending order
            indices = sorted([x[0] for x in arr])
            # Now place the sorted values into these sorted indices
            for (idx, val), pos in zip(arr, indices):
                new_arr[pos] = val
        
        # Finally check if the newly formed array is sorted
        return all(new_arr[i] <= new_arr[i+1] for i in range(len(new_arr)-1))