class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        from collections import defaultdict
        
        # Dictionary to store the indices of each value in nums
        value_indices = defaultdict(list)
        for index, value in enumerate(nums):
            value_indices[value].append(index)
        
        n = len(nums)
        result = []
        
        for query in queries:
            query_index = query
            query_value = nums[query_index]
            indices = value_indices[query_value]
            
            if len(indices) < 2:
                # If there's only one occurrence of the value, return -1
                result.append(-1)
            else:
                # Find the minimum distance to another index with the same value
                min_distance = float('inf')
                for index in indices:
                    if index != query_index:
                        # Calculate the circular distance
                        distance = min(abs(index - query_index), n - abs(index - query_index))
                        if distance < min_distance:
                            min_distance = distance
                
                result.append(min_distance)
        
        return result