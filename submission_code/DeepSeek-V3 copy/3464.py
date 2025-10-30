class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        
        # Calculate the cost of the entire array
        total_cost = 0
        sign = 1
        for num in nums:
            total_cost += num * sign
            sign *= -1
        
        # To maximize the total cost, we can split the array into individual elements
        # because the cost of each element is itself, and the sum of all elements is the maximum possible
        # However, we need to consider the alternating signs
        # So, the maximum cost is the sum of the absolute values of all elements
        max_possible = sum(abs(num) for num in nums)
        
        # Compare the total cost of the entire array with the maximum possible cost
        return max(total_cost, max_possible)