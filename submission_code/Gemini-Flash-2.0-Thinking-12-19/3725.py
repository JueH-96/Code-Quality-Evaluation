import collections
from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)

        # Helper to find nearest elements using a monotonic stack
        # Returns list of indices [left_boundary, right_boundary]
        # left_boundary[i]: index of nearest element to the left satisfying comparison_pop
        # right_boundary[i]: index of nearest element to the right satisfying comparison_boundary
        def get_nearest_elements(arr, comparison_pop, comparison_boundary):
            n = len(arr)
            left = [-1] * n
            right = [n] * n
            stack = collections.deque() # Stores indices

            # Find nearest element to the left
            for i in range(n):
                while stack and comparison_pop(arr[stack[-1]], arr[i]):
                    stack.pop()
                if stack:
                    left[i] = stack[-1]
                stack.append(i)

            # Clear stack for right pass
            stack.clear()

            # Find nearest element to the right
            for i in range(n - 1, -1, -1):
                while stack and comparison_boundary(arr[stack[-1]], arr[i]):
                    stack.pop()
                if stack:
                    right[i] = stack[-1]
                stack.append(i)

            return left, right

        # Calculate Nearest Smaller (Strictly Less Left, Less Equal Right)
        # NL[i]: index of nearest element to the left strictly LESS than arr[i]. Pop if arr[s] >= arr[i].
        # NR[i]: index of nearest element to the right LESS THAN OR EQUAL TO arr[i]. Pop if arr[s] > arr[i].
        NL, NR = get_nearest_elements(nums, lambda val_s, val_i: val_s >= val_i, lambda val_s, val_i: val_s > val_i)

        # Calculate Nearest Greater (Strictly Greater Left, Greater Equal Right)
        # NGL[i]: index of nearest element to the left STRICTLY GREATER than arr[i]. Pop if arr[s] <= arr[i].
        # NGR[i]: index of nearest element to the right GREATER THAN OR EQUAL TO arr[i]. Pop if arr[s] < arr[i].
        NGL, NGR = get_nearest_elements(nums, lambda val_s, val_i: val_s <= val_i, lambda val_s, val_i: val_s < val_i)

        total_sum = 0

        # Helper to calculate sum of max(0, i + D) for i in [A, B]
        # This sums an arithmetic progression where terms are capped at 0 from below.
        def sum_positive_arithmetic_series(A, B, D):
            if A > B:
                return 0
            
            # We need sum of max(0, i + D) for i in [A, B]
            # i + D > 0 when i > -D => i >= -D + 1
            i_pos_start = -D + 1 # Smallest integer i such that i + D >= 0

            # The range within [A, B] where i + D is positive is [max(A, i_pos_start), B]
            actual_start = max(A, i_pos_start)
            actual_end = B

            if actual_start > actual_end:
                return 0

            num_terms = actual_end - actual_start + 1
            # Sum of i from actual_start to actual_end: (first + last) * count / 2
            sum_i = (actual_start + actual_end) * num_terms // 2
            # Sum of (i + D) = Sum(i) + D * count
            total = sum_i + D * num_terms
            return total


        # Calculate contribution of each element as min and max
        for p in range(n):
            # Contribution as Minimum
            # nums[p] is min in nums[i..j] if NL[p] < i <= p <= j < NR[p]
            # Plus length constraint: j - i + 1 <= k => i >= j - k + 1
            # Combining: max(0, NL[p] + 1, j - k + 1) <= i <= p and p <= j < NR[p]
            # For a fixed p and i, j must be in [p, min(NR[p] - 1, i + k - 1)]
            # Number of valid j for fixed i, p is max(0, min(NR[p] - 1, i + k - 1) - p + 1)
            # We sum this count over valid i: i in [max(0, NL[p] + 1), p]

            min_A = max(0, NL[p] + 1) # Lower bound for i
            min_B = p                # Upper bound for i
            min_C = NR[p] - 1        # Upper bound for j based on NR

            # Function to sum over i is G(i) = max(0, min(min_C, i + k - 1) - p + 1)
            # G(i) = max(0, min(min_C - p + 1, i + k - p))
            # G(i) is piecewise linear:
            # Case 1: i + k - p < min_C - p + 1 <=> i < min_C - k + 1. Term is max(0, i + k - p)
            # Case 2: i + k - p >= min_C - p + 1 <=> i >= min_C - k + 1. Term is max(0, min_C - p + 1)

            min_i_threshold = min_C - k + 1 # Split point for i

            # Sum over i in [min_A, min(min_B, min_i_threshold - 1)]: Term is max(0, i + k - p)
            range1_start = min_A
            range1_end = min(min_B, min_i_threshold - 1)
            min_contrib_count = sum_positive_arithmetic_series(range1_start, range1_end, k - p)

            # Sum over i in [max(min_A, min_i_threshold), min_B]: Term is max(0, min_C - p + 1)
            range2_start = max(min_A, min_i_threshold)
            range2_end = min_B # = p
            
            if range2_start <= range2_end:
                constant_val = min_C - p + 1 # Value of min(min_C, i+k-1) - p + 1 when i >= i_threshold
                if constant_val > 0: # Only add if the term is positive
                    num_terms = range2_end - range2_start + 1
                    min_contrib_count += constant_val * num_terms

            total_sum += nums[p] * min_contrib_count

            # Contribution as Maximum (symmetric logic)
            # nums[p] is max in nums[i..j] if NGL[p] < i <= p <= j < NGR[p]
            # Plus length constraint: j - i + 1 <= k => i >= j - k + 1
            # Combining: max(0, NGL[p] + 1, j - k + 1) <= i <= p and p <= j < NGR[p]
            # For a fixed p and i, j must be in [p, min(NGR[p] - 1, i + k - 1)]
            # Number of valid j for fixed i, p is max(0, min(NGR[p] - 1, i + k - 1) - p + 1)
            # We sum this count over valid i: i in [max(0, NGL[p] + 1), p]

            max_A = max(0, NGL[p] + 1) # Lower bound for i
            max_B = p                # Upper bound for i
            max_C = NGR[p] - 1        # Upper bound for j based on NGR

            # Function to sum over i is H(i) = max(0, min(max_C, i + k - 1) - p + 1)
            # H(i) = max(0, min(max_C - p + 1, i + k - p))

            max_i_threshold = max_C - k + 1 # Split point for i

            # Sum over i in [max_A, min(max_B, max_i_threshold - 1)]: Term is max(0, i + k - p)
            range1_start_max = max_A
            range1_end_max = min(max_B, max_i_threshold - 1)
            max_contrib_count = sum_positive_arithmetic_series(range1_start_max, range1_end_max, k - p)

            # Sum over i in [max(max_A, max_i_threshold), max_B]: Term is max(0, max_C - p + 1)
            range2_start_max = max(max_A, max_i_threshold)
            range2_end_max = max_B # = p

            if range2_start_max <= range2_end_max:
                constant_val_max = max_C - p + 1 # Value of min(max_C, i+k-1) - p + 1 when i >= i_threshold
                if constant_val_max > 0: # Only add if the term is positive
                    num_terms = range2_end_max - range2_start_max + 1
                    max_contrib_count += constant_val_max * num_terms

            total_sum += nums[p] * max_contrib_count

        return total_sum