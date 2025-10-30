class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # Minimum cost without any rotation
        min_cost = sum(nums)
        
        # Try all possible rotations and calculate the cost
        current_cost = sum(nums)
        for k in range(1, n):
            # Rotate the array by k positions to the right
            # Calculate the cost after this rotation
            # Each rotation costs x, and we need to buy chocolates at their new positions
            # After k rotations, the chocolate originally at index i will be at index (i + k) % n
            # So we need to pay nums[i] to get chocolate originally at index i from its new position (i + k) % n
            current_cost = current_cost - nums[n-k] + nums[(n-k-1) % n] + x
            min_cost = min(min_cost, current_cost)
        
        return min_cost