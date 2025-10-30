class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        result = []
        
        for query_idx in queries:
            target_value = nums[query_idx]
            min_distance = float('inf')
            found = False
            
            # Find all indices with the same value
            for i in range(n):
                if i != query_idx and nums[i] == target_value:
                    found = True
                    # Calculate circular distance
                    # Distance going clockwise
                    clockwise = (i - query_idx) % n
                    # Distance going counterclockwise
                    counterclockwise = (query_idx - i) % n
                    
                    # Take minimum of both directions
                    distance = min(clockwise, counterclockwise)
                    min_distance = min(min_distance, distance)
            
            if found:
                result.append(min_distance)
            else:
                result.append(-1)
        
        return result