class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        if n == 1:
            return 0
        
        nums.sort()
        
        # Calculate prefix sums
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i-1] + nums[i-1]
        
        # Calculate the minimum cost
        min_cost = float('inf')
        for i in range(n):
            target = nums[i]
            left_count = i
            right_count = n - i - 1
            
            left_sum = target * left_count - (prefix_sum[i] - prefix_sum[0])
            right_sum = (prefix_sum[n] - prefix_sum[i+1]) - target * right_count
            
            # Calculate cost using cost1 and cost2
            cost = min(left_sum * cost1 + right_sum * cost1,
                       (left_sum + right_sum) * cost2 + abs(left_sum - right_sum) * (cost1 - cost2))
            
            min_cost = min(min_cost, cost)
        
        return min_cost % MOD