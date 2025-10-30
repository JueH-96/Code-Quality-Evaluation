from typing import List
from collections import defaultdict

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        # Dictionary to store the indices of each number in nums
        index_map = defaultdict(list)
        
        # Fill the index_map with indices for each number
        for index, value in enumerate(nums):
            index_map[value].append(index)
        
        # Prepare the result list
        result = []
        n = len(nums)
        
        # Process each query
        for query in queries:
            value = nums[query]
            indices = index_map[value]
            min_distance = float('inf')
            
            # If there are less than 2 indices, return -1
            if len(indices) < 2:
                result.append(-1)
                continue
            
            # Find the minimum distance in the circular array
            for i in range(len(indices)):
                current_index = indices[i]
                next_index = indices[(i + 1) % len(indices)]
                
                # Calculate the distance in both directions
                distance = min((next_index - current_index) % n, (current_index - next_index) % n)
                min_distance = min(min_distance, distance)
            
            result.append(min_distance)
        
        return result