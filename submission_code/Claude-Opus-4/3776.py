class Solution:
    def minCost(self, nums: List[int]) -> int:
        from functools import lru_cache
        
        @lru_cache(None)
        def dp(remaining):
            # Convert tuple back to list for easier manipulation
            nums_list = list(remaining)
            n = len(nums_list)
            
            # Base case: if less than 3 elements, remove all
            if n < 3:
                return max(nums_list) if nums_list else 0
            
            # Try all possible pairs from the first 3 elements
            min_cost = float('inf')
            
            # Choose 2 elements from indices 0, 1, 2
            for i in range(3):
                for j in range(i + 1, 3):
                    # Cost of removing elements at indices i and j
                    cost = max(nums_list[i], nums_list[j])
                    
                    # Create new array after removing elements i and j
                    new_nums = []
                    for k in range(n):
                        if k != i and k != j:
                            new_nums.append(nums_list[k])
                    
                    # Recursively calculate the cost for remaining elements
                    total_cost = cost + dp(tuple(new_nums))
                    min_cost = min(min_cost, total_cost)
            
            return min_cost
        
        return dp(tuple(nums))