class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        def min_increments_for_target(num: int, candidates: List[int]) -> int:
            # Find minimum increments needed to make any candidate a multiple of target
            min_ops = float('inf')
            for candidate in candidates:
                if candidate >= num:
                    # Calculate how many increments needed to reach next multiple
                    quotient = candidate // num
                    next_multiple = (quotient + 1) * num
                    ops = next_multiple - candidate
                    min_ops = min(min_ops, ops)
                else:
                    # Calculate how many increments needed to reach next multiple
                    quotient = candidate // num
                    next_multiple = (quotient + 1) * num
                    ops = next_multiple - candidate
                    min_ops = min(min_ops, ops)
            return min_ops
        
        total_ops = 0
        nums = sorted(nums)  # Sort for potential optimization
        
        # For each target number
        for t in target:
            # Get minimum increments needed for this target
            ops = min_increments_for_target(t, nums)
            if ops == float('inf'):
                return -1  # Impossible case
            total_ops += ops
            
        return total_ops