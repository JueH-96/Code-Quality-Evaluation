class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        if n == 1:
            return 0
        
        max_val = max(nums)
        
        # Calculate total increments needed
        total_increments = sum(max_val - num for num in nums)
        
        if total_increments == 0:
            return 0
        
        # If cost2 >= 2 * cost1, it's always better to use operation 1
        if cost2 >= 2 * cost1:
            return (total_increments * cost1) % MOD
        
        # Find the minimum value and its deficit
        min_val = min(nums)
        max_deficit = max_val - min_val
        
        # Try different target values (max_val and max_val + 1)
        min_cost = float('inf')
        
        for target in range(max_val, max_val + 2):
            total_ops = sum(target - num for num in nums)
            max_single_deficit = target - min_val
            
            # If the maximum deficit is more than half of total operations,
            # some operations must be done using cost1
            if max_single_deficit * 2 > total_ops:
                # We can pair up to (total_ops - max_single_deficit) operations
                pairs = total_ops - max_single_deficit
                singles = total_ops - 2 * pairs
                cost = pairs * cost2 + singles * cost1
            else:
                # We can pair all operations optimally
                if total_ops % 2 == 0:
                    cost = (total_ops // 2) * cost2
                else:
                    # One operation must be single
                    cost = (total_ops // 2) * cost2 + cost1
            
            min_cost = min(min_cost, cost)
        
        return min_cost % MOD