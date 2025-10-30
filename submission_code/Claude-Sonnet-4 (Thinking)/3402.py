class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        
        n = len(nums)
        max_val = max(nums)
        min_val = min(nums)
        sum_nums = sum(nums)
        
        def cost_for_target(target):
            total = n * target - sum_nums
            max_increase = target - min_val
            
            if cost2 >= 2 * cost1:
                return total * cost1
            else:
                max_op2 = min(total // 2, total - max_increase)
                return max_op2 * cost2 + (total - 2 * max_op2) * cost1
        
        min_cost = cost_for_target(max_val)
        
        # Check if it's beneficial to increase target beyond max_val
        if cost2 < 2 * cost1 and n > 2:
            # In some cases, increasing target can balance the increases better
            # and allow more operation 2's to be used
            slope_case_imbalanced = (n - 1) * cost2 - (n - 2) * cost1
            if slope_case_imbalanced < 0:
                # Cost decreases when target increases in the imbalanced case
                # Check around the transition point where increases become balanced
                t_transition = (sum_nums - 2 * min_val) // (n - 2)
                for t in range(max_val + 1, min(t_transition + 5, max_val + 100000)):
                    min_cost = min(min_cost, cost_for_target(t))
        
        return min_cost % MOD