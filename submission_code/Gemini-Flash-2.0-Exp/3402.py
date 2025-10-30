class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        n = len(nums)
        min_cost = float('inf')
        
        for target in sorted(list(set(nums))):
            cost = 0
            for num in nums:
                diff = target - num
                if diff > 0:
                    cost += min(diff * cost1, (diff // 2) * cost2 + (diff % 2) * cost1)
            min_cost = min(min_cost, cost)
            
        nums.sort()
        min_val = nums[0]
        max_val = nums[-1]
        
        for target in range(min_val, max_val + 1):
            cost = 0
            for num in nums:
                diff = target - num
                if diff > 0:
                    cost += min(diff * cost1, (diff // 2) * cost2 + (diff % 2) * cost1)
            min_cost = min(min_cost, cost)
            
        return min_cost % (10**9 + 7)