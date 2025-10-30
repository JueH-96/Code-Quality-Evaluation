class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # If cost2/2 is less than cost1, always use cost2 operation
        if cost2 < 2 * cost1:
            cost1 = cost2 // 2
        
        # Find min cost to make all elements equal to target
        def getCost(target):
            cost = 0
            remaining = 0
            
            for num in nums:
                diff = abs(target - num)
                # Use as many cost2 operations as possible
                pairs = diff // 2
                cost = (cost + pairs * cost2) % MOD
                remaining += diff % 2
            
            # Use cost1 for remaining single increments
            cost = (cost + (remaining + 1) // 2 * cost2 + (remaining % 2) * cost1) % MOD
            return cost
        
        # Binary search for optimal target value
        left = min(nums)
        right = max(nums)
        
        while right - left > 2:
            mid = (left + right) // 2
            cost_mid = getCost(mid)
            cost_mid_plus = getCost(mid + 1)
            
            if cost_mid <= cost_mid_plus:
                right = mid
            else:
                left = mid
                
        # Check all possible values near the optimal point
        result = float('inf')
        for target in range(left, right + 1):
            result = min(result, getCost(target))
            
        return result