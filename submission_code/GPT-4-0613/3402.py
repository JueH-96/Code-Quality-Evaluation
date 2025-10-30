class Solution:
    def minCostToEqualizeArray(self, nums, cost1, cost2):
        nums.sort()
        n = len(nums)
        prefix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        res = float('inf')
        for i in range(n):
            total = prefix[n] - prefix[i+1]
            need = nums[i] * (n-i-1) - total
            cost = need * min(cost1, cost2 * 2)
            if i > 0:
                total = prefix[i] - prefix[0]
                need = nums[i] * i - total
                cost += need * min(cost1, cost2 * 2)
            res = min(res, cost)
        return res % (10**9 + 7)