class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        # Create a dictionary to store indices for each number
        num_indices = {}
        
        # Store all indices for each number
        for i in range(n):
            if nums[i] not in num_indices:
                num_indices[nums[i]] = []
            num_indices[nums[i]].append(i)
        
        result = []
        
        for query_idx in queries:
            query_num = nums[query_idx]
            
            # If the number appears only once, it has no other occurrence
            if len(num_indices[query_num]) == 1:
                result.append(-1)
                continue
            
            # Find minimum circular distance
            min_distance = float('inf')
            for idx in num_indices[query_num]:
                if idx != query_idx:
                    # Calculate both clockwise and counterclockwise distances
                    dist = abs(idx - query_idx)
                    circular_dist = min(dist, n - dist)
                    min_distance = min(min_distance, circular_dist)
            
            result.append(min_distance)
        
        return result