class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        from collections import defaultdict
        
        # Create a dictionary to store the indices of each number in nums
        index_map = defaultdict(list)
        n = len(nums)
        
        # Populate the index_map with indices for each number
        for i, num in enumerate(nums):
            index_map[num].append(i)
        
        # Prepare the result list
        result = []
        
        # Process each query
        for query in queries:
            num = nums[query]
            indices = index_map[num]
            
            # If there is only one occurrence of the number, return -1
            if len(indices) == 1:
                result.append(-1)
                continue
            
            # Find the minimum circular distance to another index with the same value
            min_distance = float('inf')
            for idx in indices:
                if idx != query:
                    # Calculate the circular distance
                    distance = min(abs(idx - query), n - abs(idx - query))
                    min_distance = min(min_distance, distance)
            
            result.append(min_distance)
        
        return result