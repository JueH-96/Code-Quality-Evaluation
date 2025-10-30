class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # Initialize the minimum cost for each type as the original cost
        min_cost = nums.copy()
        total_min_cost = sum(min_cost)
        
        # We can perform up to n-1 operations since after n operations, the types cycle back
        for op in range(1, n):
            current_cost = op * x
            # Update the min_cost for each type considering the current operation
            for i in range(n):
                # The type after op operations is (i + op) % n
                new_type = (i + op) % n
                if nums[new_type] < min_cost[i]:
                    min_cost[i] = nums[new_type]
            # Calculate the total cost for the current number of operations
            current_total = current_cost + sum(min_cost)
            if current_total < total_min_cost:
                total_min_cost = current_total
        
        return total_min_cost