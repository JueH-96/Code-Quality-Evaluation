class Solution:
    def minCost(self, nums: List[int]) -> int:
        # Use memoization to store already computed results
        @cache
        def dp(start: int, end: int) -> int:
            # If there are 2 or fewer elements remaining
            if end - start <= 2:
                if end - start == 0:  # Empty array
                    return 0
                # Return max of remaining elements
                return max(nums[start:end])
            
            # Initialize minimum cost with a large value
            min_cost = float('inf')
            
            # Try all possible combinations of removing 2 elements from first 3
            for i in range(start, min(start + 3, end)):
                for j in range(i + 1, min(start + 3, end)):
                    # Current operation cost is max of two elements being removed
                    current_cost = max(nums[i], nums[j])
                    
                    # Create new array after removing elements at i and j
                    # We can simulate this by copying remaining elements
                    remaining = nums[start:i] + nums[i+1:j] + nums[j+1:end]
                    next_cost = dp(0, len(remaining))
                    
                    # Update minimum cost if current combination is better
                    min_cost = min(min_cost, current_cost + next_cost)
            
            return min_cost
        
        return dp(0, len(nums))