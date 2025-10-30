class Solution:
    def minCost(self, nums: List[int]) -> int:
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(arr):
            # Base case: if there are 0-2 elements left, remove all of them
            if len(arr) <= 2:
                return max(arr) if arr else 0
            
            min_cost = float('inf')
            
            # Try all possible pairs from the first three elements
            for i in range(min(3, len(arr))):
                for j in range(i+1, min(3, len(arr))):
                    # Calculate cost for this removal
                    cost = max(arr[i], arr[j])
                    
                    # Create new tuple with elements i and j removed
                    remaining = arr[:i] + arr[i+1:j] + arr[j+1:]
                    
                    # Calculate total cost with this choice
                    total_cost = cost + dp(remaining)
                    min_cost = min(min_cost, total_cost)
            
            return min_cost
        
        return dp(tuple(nums))