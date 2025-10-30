class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        
        # Find the target as the maximum value in the array
        target = max(nums)
        
        # Calculate the total increases needed and the number of elements that need increases
        total_increases = 0
        k = 0
        for num in nums:
           差 = target - num
            if 差 > 0:
                total_increases += 差
                k += 1
        
        # Determine the number of two-increase operations to use
        if cost2 < 2 * cost1:
            x = min(total_increases // 2, k)
        else:
            x = 0
        
        # Calculate the remaining increases needed
        y = total_increases - 2 * x
        
        # Calculate the total cost
        cost = (x * cost2 + y * cost1) % MOD
        
        return cost