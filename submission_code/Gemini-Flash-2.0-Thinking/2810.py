class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_total_cost = float('inf')

        for num_operations in range(n):  # We only need to check up to n-1 operations
            current_cost = num_operations * x
            collection_costs = [float('inf')] * n  # Cost to collect each type

            # After num_operations, the chocolate at index i has type (i + num_operations) % n

            for target_type in range(n):
                for i in range(n):
                    current_type = (i + num_operations) % n
                    if current_type == target_type:
                        collection_costs[target_type] = min(collection_costs[target_type], nums[i])

            current_cost += sum(collection_costs)
            min_total_cost = min(min_total_cost, current_cost)

        return min_total_cost