class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_cost = float('inf')

        # Try all possible rotations
        for rotation in range(n):
            current_cost = 0
            for i in range(n):
                # Calculate the new index after rotation
                new_index = (i + rotation) % n
                current_cost += nums[new_index]
            # Add the cost of rotations
            current_cost += rotation * x
            min_cost = min(min_cost, current_cost)

        return min_cost