class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_cost = float('inf')
        for start in range(n):
            cost = 0
            current_nums = nums[:]
            current_cost = 0
            for i in range(n):
                index = (start + i) % n
                cost += current_nums[index]
            
            
            min_cost = min(min_cost, cost)
            
            
            for num_ops in range(1,n):
                cost_with_ops = num_ops * x
                current_nums = nums[:]
                current_cost = 0
                for i in range(n):
                    index = (start + i + num_ops) % n
                    current_cost += current_nums[index]
                
                min_cost = min(min_cost, current_cost + cost_with_ops)

        return min_cost