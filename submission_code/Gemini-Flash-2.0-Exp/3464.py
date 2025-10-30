class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        def calculate_cost(l, r):
            cost = 0
            for i in range(l, r + 1):
                cost += nums[i] * (1 if (i - l) % 2 == 0 else -1)
            return cost

        if n == 1:
            return calculate_cost(0, n - 1)

        max_cost = calculate_cost(0, n - 1)
        
        for i in range(1, 1 << (n - 1)):
            current_cost = 0
            splits = []
            for j in range(n - 1):
                if (i >> j) & 1:
                    splits.append(j)
            
            if not splits:
                current_cost = calculate_cost(0, n - 1)
            else:
                current_cost += calculate_cost(0, splits[0])
                for k in range(len(splits) - 1):
                    current_cost += calculate_cost(splits[k] + 1, splits[k+1])
                current_cost += calculate_cost(splits[-1] + 1, n - 1)
            
            max_cost = max(max_cost, current_cost)
            
        return max_cost