from collections import defaultdict

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        
        # Create a mapping of each value to all its indices
        value_to_indices = defaultdict(list)
        for i, num in enumerate(nums):
            value_to_indices[num].append(i)
        
        answer = []
        for query in queries:
            value = nums[query]
            indices = value_to_indices[value]
            
            if len(indices) == 1:  # Only the queried index has this value
                answer.append(-1)
            else:
                min_distance = float('inf')
                for idx in indices:
                    if idx != query:  # Skip the queried index itself
                        # Calculate circular distance (minimum of clockwise and counterclockwise)
                        distance = min(abs(idx - query), n - abs(idx - query))
                        min_distance = min(min_distance, distance)
                answer.append(min_distance)
        
        return answer