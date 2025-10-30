class Solution:
    def minCost(self, nums: List[int]) -> int:
        from functools import lru_cache
        
        @lru_cache(None)
        def dp(state):
            # Convert tuple back to list for easier manipulation
            arr = list(state)
            n = len(arr)
            
            # Base case: if fewer than 3 elements, remove all
            if n <= 2:
                return max(arr) if arr else 0
            
            min_cost = float('inf')
            
            # Try all possible pairs from first 3 elements
            for i in range(3):
                for j in range(i + 1, 3):
                    # Cost of removing arr[i] and arr[j]
                    cost = max(arr[i], arr[j])
                    
                    # Create new array after removing elements
                    new_arr = []
                    for k in range(n):
                        if k != i and k != j:
                            new_arr.append(arr[k])
                    
                    # Recursively solve for remaining array
                    total_cost = cost + dp(tuple(new_arr))
                    min_cost = min(min_cost, total_cost)
            
            return min_cost
        
        return dp(tuple(nums))