class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        max_value = max(nums)
        n = len(nums)
        
        # Calculate the cost to make all elements equal to max_value
        cost_to_max = 0
        for num in nums:
            cost_to_max += (max_value - num) * cost1
            cost_to_max %= MOD
        
        # If cost2 >= 2 * cost1, then it's never beneficial to use the cost2 operation
        if cost2 >= 2 * cost1:
            return cost_to_max
        
        # Try to minimize the cost by considering making all elements equal to values > max_value
        min_cost = cost_to_max
        additional_steps = 0
        
        # We only need to consider increasing the target value up to the point where
        # the cost of using pairs (cost2) exceeds using individual increments (cost1)
        while True:
            # Increase the target value by 1
            additional_steps += 1
            # Cost to move all elements to this new target using the most efficient method
            new_target_cost = (additional_steps * n * cost1) % MOD
            
            # Check if using pairs can reduce the cost
            if n % 2 == 0:
                # If n is even, we can pair all elements
                pair_cost = (additional_steps * (n // 2) * cost2) % MOD
            else:
                # If n is odd, one element will be left out every time
                pair_cost = ((additional_steps * (n // 2) * cost2) + (additional_steps * cost1)) % MOD
            
            current_cost = (cost_to_max + min(pair_cost, new_target_cost)) % MOD
            if current_cost < min_cost:
                min_cost = current_cost
            else:
                # If the cost starts increasing, further increases won't be beneficial
                break
        
        return min_cost