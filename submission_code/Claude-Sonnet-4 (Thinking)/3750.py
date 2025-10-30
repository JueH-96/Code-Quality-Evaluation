class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        
        # Group indices by value
        value_to_indices = {}
        for i, val in enumerate(nums):
            if val not in value_to_indices:
                value_to_indices[val] = []
            value_to_indices[val].append(i)
        
        answer = []
        
        for query_idx in queries:
            target_value = nums[query_idx]
            indices_with_same_value = value_to_indices[target_value]
            
            min_distance = float('inf')
            
            for j in indices_with_same_value:
                if j != query_idx:
                    # Calculate circular distance
                    distance = min((j - query_idx) % n, (query_idx - j) % n)
                    min_distance = min(min_distance, distance)
            
            if min_distance == float('inf'):
                answer.append(-1)
            else:
                answer.append(min_distance)
        
        return answer