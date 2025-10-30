class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        
        def calculate_cost(target):
            cost = 0
            for num in nums:
                diff = abs(num - target)
                if cost1 * 2 <= cost2:
                    cost += diff * cost1
                else:
                    cost += (diff // 2) * cost2 + (diff % 2) * cost1
            return cost
        
        left, right = min(nums), max(nums)
        while left < right:
            mid = (left + right) // 2
            cost_mid = calculate_cost(mid)
            cost_mid_next = calculate_cost(mid + 1)
            if cost_mid_next < cost_mid:
                left = mid + 1
            else:
                right = mid
        
        return calculate_cost(left) % MOD