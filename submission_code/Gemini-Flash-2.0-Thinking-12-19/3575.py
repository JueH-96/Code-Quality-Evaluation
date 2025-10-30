class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        max_val = 0

        def calculate_value(subsequence):
            n = len(subsequence)
            if n != 2 * k:
                return 0
            first_half_or = 0
            for i in range(k):
                first_half_or |= subsequence[i]
            second_half_or = 0
            for i in range(k, 2 * k):
                second_half_or |= subsequence[i]
            return first_half_or ^ second_half_or

        def find_max_value(index, current_subsequence):
            nonlocal max_val
            if len(current_subsequence) == 2 * k:
                current_value = calculate_value(current_subsequence)
                max_val = max(max_val, current_value)
                return

            if index == len(nums):
                return
            
            # Include nums[index]
            find_max_value(index + 1, current_subsequence + [nums[index]])
            # Exclude nums[index]
            find_max_value(index + 1, current_subsequence)

        find_max_value(0, [])
        
        # Optimization: Early exit if we have selected enough elements
        def find_max_value_optimized(index, current_subsequence):
            nonlocal max_val
            if len(current_subsequence) == 2 * k:
                current_value = calculate_value(current_subsequence)
                max_val = max(max_val, current_value)
                return

            if index == len(nums):
                return
            
            # Include nums[index]
            find_max_value_optimized(index + 1, current_subsequence + [nums[index]])
            # Exclude nums[index]
            if len(nums) - index + len(current_subsequence) >= 2 * k: # Optimization to prune branches
                find_max_value_optimized(index + 1, current_subsequence)

        max_val = 0
        find_max_value_optimized(0, [])
        return max_val