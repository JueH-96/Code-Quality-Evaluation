class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        total_cost = 0
        current_length = 0
        
        for num in nums:
            if current_length % 2 == 1 and num > 0:
                # Start a new subarray
                total_cost += num
                current_length = 1
            else:
                # Extend the current subarray
                if current_length % 2 == 0:
                    total_cost += num
                else:
                    total_cost -= num
                current_length += 1
        
        return total_cost