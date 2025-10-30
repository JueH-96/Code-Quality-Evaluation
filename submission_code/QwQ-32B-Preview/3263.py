class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # The first part always starts with nums[0]
        first = nums[0]
        
        # Get the remaining array starting from the second element
        remaining = nums[1:]
        
        # Sort the remaining array
        remaining_sorted = sorted(remaining)
        
        # Sum the two smallest elements from the sorted remaining array
        min_sum = remaining_sorted[0] + remaining_sorted[1]
        
        # Total cost is the sum of the first element and the two smallest in the remaining array
        total_cost = first + min_sum
        
        return total_cost