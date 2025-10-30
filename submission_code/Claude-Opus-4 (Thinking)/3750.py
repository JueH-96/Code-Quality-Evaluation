class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        
        # Create a map from value to list of indices
        value_to_indices = {}
        for i, num in enumerate(nums):
            if num not in value_to_indices:
                value_to_indices[num] = []
            value_to_indices[num].append(i)
        
        answer = []
        
        for query_idx in queries:
            target_value = nums[query_idx]
            indices = value_to_indices[target_value]
            
            if len(indices) == 1:
                # Only one occurrence of this value
                answer.append(-1)
            else:
                min_distance = float('inf')
                for idx in indices:
                    if idx != query_idx:
                        # Calculate circular distance
                        direct_distance = abs(query_idx - idx)
                        circular_distance = min(direct_distance, n - direct_distance)
                        min_distance = min(min_distance, circular_distance)
                
                answer.append(min_distance)
        
        return answer