class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        n = len(nums)
        mod = 10**9 + 7

        def get_cost(target):
            cost = 0
            inc = 0
            dec = 0
            for num in nums:
                if num < target:
                    inc += target - num
                elif num > target:
                    dec += num - target
            if cost1 < cost2:
                common = min(inc, dec)
                cost = (common * cost2) + (inc - common) * cost1 + (dec - common) * cost1
            else:
                cost = inc * cost1 + dec * cost1
                
            return cost

        left = min(nums)
        right = max(nums)
        ans = float('inf')

        while left <= right:
            mid1 = left + (right - left) // 3
            mid2 = right - (right - left) // 3
            cost1 = get_cost(mid1)
            cost2 = get_cost(mid2)
            ans = min(ans, cost1, cost2)
            if cost1 < cost2:
                right = mid2 - 1
            else:
                left = mid1 + 1
        
        return ans