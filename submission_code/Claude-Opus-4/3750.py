class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        # Create a mapping of values to their indices
        value_to_indices = {}
        for i, num in enumerate(nums):
            if num not in value_to_indices:
                value_to_indices[num] = []
            value_to_indices[num].append(i)
        
        n = len(nums)
        result = []
        
        for query_idx in queries:
            value = nums[query_idx]
            indices = value_to_indices[value]
            
            # If there's only one index with this value, return -1
            if len(indices) == 1:
                result.append(-1)
            else:
                min_distance = float('inf')
                
                # Check distance to all other indices with the same value
                for idx in indices:
                    if idx != query_idx:
                        # Calculate circular distance
                        linear_distance = abs(idx - query_idx)
                        circular_distance = min(linear_distance, n - linear_distance)
                        min_distance = min(min_distance, circular_distance)
                
                result.append(min_distance)
        
        return result