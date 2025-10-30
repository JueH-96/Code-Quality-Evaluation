from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        memo = {}
        
        def solve(current_types, collected_mask):
            types_tuple = tuple(current_types)
            if collected_mask == (1 << n) - 1:
                return 0
            if (types_tuple, collected_mask) in memo:
                return memo[(types_tuple, collected_mask)]
            
            # Option 1: Perform operation
            next_types = [(t + 1) % n for t in current_types]
            cost1 = x + solve(next_types, collected_mask)
            
            # Option 2: Collect a type
            min_cost_collect = float('inf')
            for type_to_collect in range(n):
                if not (collected_mask & (1 << type_to_collect)):
                    indices_of_type = [i for i, t in enumerate(current_types) if t == type_to_collect]
                    if indices_of_type:
                        best_index = -1
                        min_price = float('inf')
                        for index in indices_of_type:
                            if nums[index] < min_price:
                                min_price = nums[index]
                                best_index = index
                        cost2 = nums[best_index] + solve(current_types, collected_mask | (1 << type_to_collect))
                        min_cost_collect = min(min_cost_collect, cost2)
                        
            result = min(cost1, min_cost_collect)
            memo[(types_tuple, collected_mask)] = result
            return result
            
        initial_types = list(range(n))
        initial_collected_mask = 0
        return solve(initial_types, initial_collected_mask)