class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        if n == 1:
            return 0
        
        min_val = min(nums)
        max_val = max(nums)
        total_sum = sum(nums)
        
        if min_val == max_val:
            return 0
        
        def calculate_cost(target):
            total_ops = n * target - total_sum
            max_diff = target - min_val
            
            # Cost using only operation 1
            cost_only_op1 = cost1 * total_ops
            
            # If operation 2 is not beneficial
            if cost2 >= 2 * cost1:
                return cost_only_op1
            
            # Calculate how many operation 2's we can use
            if max_diff * 2 <= total_ops:
                # We can pair all operations
                num_op2 = total_ops // 2
                num_op1 = total_ops % 2
            else:
                # We can't pair all operations of the min element
                num_op2 = total_ops - max_diff
                num_op1 = 2 * max_diff - total_ops
            
            cost_with_op2 = num_op2 * cost2 + num_op1 * cost1
            
            return min(cost_only_op1, cost_with_op2)
        
        # For n = 2, the optimal target is usually max_val
        if n == 2:
            return calculate_cost(max_val) % MOD
        
        # For n > 2, find the threshold where we can pair all operations
        # This happens when max_diff = total_ops / 2
        # Solving: t - min_val = (n * t - total_sum) / 2
        # We get: t = (total_sum - 2 * min_val) / (n - 2)
        
        min_cost = calculate_cost(max_val)
        
        if cost2 < 2 * cost1:
            # Calculate threshold
            if total_sum > 2 * min_val:
                threshold = (total_sum - 2 * min_val + n - 3) // (n - 2)  # Ceiling division
                
                # Check targets around the threshold
                for t in range(max(max_val + 1, threshold - 1), threshold + 2):
                    cost = calculate_cost(t)
                    min_cost = min(min_cost, cost)
            
            # Also check a few more targets to be safe
            for t in range(max_val + 1, max_val + 11):
                cost = calculate_cost(t)
                if cost >= min_cost:
                    break
                min_cost = cost
        
        return min_cost % MOD