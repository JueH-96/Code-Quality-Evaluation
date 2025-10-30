class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        total_cost = 0
        
        # Collect the chocolates of the initial types
        for i in range(n):
            total_cost += nums[i]
        
        # Perform the operation as many times as needed
        while True:
            min_cost = float('inf')
            for i in range(n):
                next_type = (i + 1) % n
                min_cost = min(min_cost, nums[i] + nums[next_type])
            
            if min_cost >= x:
                break
            
            total_cost += x
            
            # Update the chocolate types
            for i in range(n):
                next_type = (i + 1) % n
                nums[i] = nums[next_type]
        
        return total_cost