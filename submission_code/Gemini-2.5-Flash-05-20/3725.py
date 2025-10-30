import collections
from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_sum = 0

        # --- Step 1: Precompute left_smaller/greater and right_smaller/greater indices ---
        
        # Initialize with sentinels
        left_smaller = [-1] * n
        right_smaller = [n] * n
        left_greater = [-1] * n
        right_greater = [n] * n

        stack = collections.deque() # For smaller elements (to find previous smaller)
        for i in range(n):
            # Pop elements from stack that are greater than or equal to nums[i]
            # This ensures left_smaller[i] points to the previous STRICTLY smaller element
            while stack and nums[stack[-1]] >= nums[i]:
                stack.pop()
            if stack:
                left_smaller[i] = stack[-1]
            stack.append(i)
        
        stack = collections.deque() # For smaller elements (to find next smaller or equal)
        for i in range(n - 1, -1, -1):
            # Pop elements from stack that are strictly greater than nums[i]
            # This ensures right_smaller[i] points to the next SMALLER OR EQUAL element
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            if stack:
                right_smaller[i] = stack[-1]
            stack.append(i)

        stack = collections.deque() # For greater elements (to find previous greater)
        for i in range(n):
            # Pop elements from stack that are smaller than or equal to nums[i]
            # This ensures left_greater[i] points to the previous STRICTLY greater element
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                left_greater[i] = stack[-1]
            stack.append(i)
        
        stack = collections.deque() # For greater elements (to find next greater or equal)
        for i in range(n - 1, -1, -1):
            # Pop elements from stack that are strictly smaller than nums[i]
            # This ensures right_greater[i] points to the next GREATER OR EQUAL element
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            if stack:
                right_greater[i] = stack[-1]
            stack.append(i)

        # Helper to calculate sum of arithmetic series: sum(x) for x from start to end
        def sum_arithmetic(start, end):
            if start > end:
                return 0
            # Sum of an arithmetic progression: (number of terms) * (first term + last term) / 2
            return (end - start + 1) * (start + end) // 2

        # --- Step 2: Calculate contributions for each element as min and max ---
        for i in range(n):
            val = nums[i]

            # Calculate contribution as Minimum
            
            # Subarray starts L in [left_smaller[i] + 1, i]
            # Subarray ends R in [i, right_smaller[i] - 1]
            # And R - L + 1 <= k (Length constraint) => R <= L + k - 1

            L_range_start_min = left_smaller[i] + 1
            R_range_end_min = right_smaller[i] - 1
            
            # Effective L start, considering both its range as min and the k-length window constraint
            L_effective_start_min = max(L_range_start_min, i - k + 1)

            # Split point for L where L + k - 1 starts exceeding R_range_end_min
            split_point_L_min = R_range_end_min - k + 1

            current_min_contribution = 0

            # Part 1: L values where R is limited by L + k - 1 (length constraint dominates)
            # count_R_for_L = (L + k - 1) - i + 1 = L + (k - i)
            L_part1_start_min = L_effective_start_min
            L_part1_end_min = min(i, split_point_L_min - 1)
            
            if L_part1_start_min <= L_part1_end_min:
                # Sum (L + (k - i)) for L from L_part1_start_min to L_part1_end_min
                # This is an arithmetic series: (start_val + end_val) * count / 2
                start_val_series = L_part1_start_min + (k - i)
                end_val_series = L_part1_end_min + (k - i)
                current_min_contribution += val * sum_arithmetic(start_val_series, end_val_series)
            
            # Part 2: L values where R is limited by R_range_end_min (influence range dominates)
            # count_R_for_L = R_range_end_min - i + 1 (constant)
            L_part2_start_min = max(L_effective_start_min, split_point_L_min)
            L_part2_end_min = i
            
            if L_part2_start_min <= L_part2_end_min:
                count_R_for_L_const = R_range_end_min - i + 1
                num_L_values = L_part2_end_min - L_part2_start_min + 1
                current_min_contribution += val * count_R_for_L_const * num_L_values
            
            total_sum += current_min_contribution

            # Calculate contribution as Maximum (Symmetric logic)
            
            L_range_start_max = left_greater[i] + 1
            R_range_end_max = right_greater[i] - 1
            
            L_effective_start_max = max(L_range_start_max, i - k + 1)
            split_point_L_max = R_range_end_max - k + 1

            current_max_contribution = 0

            L_part1_start_max = L_effective_start_max
            L_part1_end_max = min(i, split_point_L_max - 1)
            
            if L_part1_start_max <= L_part1_end_max:
                start_val_series = L_part1_start_max + (k - i)
                end_val_series = L_part1_end_max + (k - i)
                current_max_contribution += val * sum_arithmetic(start_val_series, end_val_series)
            
            L_part2_start_max = max(L_effective_start_max, split_point_L_max)
            L_part2_end_max = i
            
            if L_part2_start_max <= L_part2_end_max:
                count_R_for_L_const = R_range_end_max - i + 1
                num_L_values = L_part2_end_max - L_part2_start_max + 1
                current_max_contribution += val * count_R_for_L_const * num_L_values
            
            total_sum += current_max_contribution
        
        return total_sum