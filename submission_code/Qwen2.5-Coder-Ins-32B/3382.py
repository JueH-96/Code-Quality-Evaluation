from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        # Dictionary to store the indices of each number
        index_map = defaultdict(list)
        
        # Populate the dictionary with indices
        for i, num in enumerate(nums):
            index_map[num].append(i)
        
        count = 0
        
        # Calculate the number of valid subarrays for each unique number
        for indices in index_map.values():
            n = len(indices)
            # For each pair of indices (i, j) where i <= j, it forms a valid subarray
            count += n * (n + 1) // 2
        
        return count