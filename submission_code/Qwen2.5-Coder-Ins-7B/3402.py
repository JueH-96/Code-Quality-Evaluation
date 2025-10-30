class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        def cost(target):
            total_cost = 0
            for num in nums:
                diff = abs(num - target)
                total_cost += diff * cost1
                total_cost += (diff - 1) * cost2
            return total_cost
        
        left, right = min(nums), max(nums)
        while left < right:
            mid = (left + right) // 2
            if cost(mid) < cost(mid + 1):
                right = mid
            else:
                left = mid + 1
        
        return cost(left) % MOD