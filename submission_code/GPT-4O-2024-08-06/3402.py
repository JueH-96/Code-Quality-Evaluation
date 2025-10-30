class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        
        # Sort the array to easily find the median
        nums.sort()
        n = len(nums)
        
        # Find the median
        if n % 2 == 1:
            target = nums[n // 2]
        else:
            target = nums[n // 2 - 1]
        
        # Calculate the cost to make all elements equal to the target
        total_cost = 0
        for num in nums:
            diff = abs(num - target)
            if cost1 * 2 <= cost2:
                # It's cheaper to use single increments
                total_cost += diff * cost1
            else:
                # Use pair increments as much as possible
                pairs = diff // 2
                singles = diff % 2
                total_cost += pairs * cost2 + singles * cost1
        
        return total_cost % MOD