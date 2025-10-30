from typing import List
import math

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)

        # Find max difference between adjacent non--1 elements
        max_fixed_diff = 0
        last_non_neg = None
        has_missing = False
        non_missing_values = []
        for x in nums:
            if x != -1:
                non_missing_values.append(x)
                if last_non_neg is not None:
                    max_fixed_diff = max(max_fixed_diff, abs(x - last_non_neg))
                last_non_neg = x
            else:
                has_missing = True

        # If no -1s, answer is max_fixed_diff
        if not has_missing:
             return max_fixed_diff
             
        # If all elements are -1, answer is 0
        if not non_missing_values:
             return 0

        # Find all unique non--1 elements adjacent to any -1s
        adj_non_missing = set()
        for i in range(n):
            if nums[i] == -1:
                if i > 0 and nums[i-1] != -1:
                    adj_non_missing.add(nums[i-1])
                if i < n - 1 and nums[i+1] != -1:
                    adj_non_missing.add(nums[i+1])

        # If there are -1s but no non-missing values adjacent to them,
        # it means -1s are only at the ends and there's a gap of non-missing values in between.
        # e.g., [-1, -1, 5, 6, -1]. max_fixed_diff is handled. The end -1s can be anything.
        # However, the problem statement implies adjacent means index i-1 or i+1.
        # If nums = [-1, -1, 5, -1, -1], P = {5}. min_P=5, max_P=5.
        # If P is empty here, it means no non-missing values at all (handled)
        # or non-missing values are not adjacent to any -1s (not possible by construction of P).
        # So if has_missing is true and non_missing_values is true, P must be non-empty.
        # We can skip the P empty check if non_missing_values is not empty.

        min_P = min(adj_non_missing)
        max_P = max(adj_non_missing)


        # can(D) function: Check if max difference D is possible
        def can(D):
            if max_fixed_diff > D:
                return False

            # Collect required intervals for each -1 position
            required_intervals = []
            for i in range(n):
                if nums[i] == -1:
                    L_i = 1
                    R_i = 10**9 + 7 # Use a large enough value

                    # Check left neighbor
                    if i > 0 and nums[i-1] != -1:
                        neighbor = nums[i-1]
                        L_i = max(L_i, neighbor - D)
                        R_i = min(R_i, neighbor + D)

                    # Check right neighbor
                    if i < n - 1 and nums[i+1] != -1:
                        neighbor = nums[i+1]
                        L_i = max(L_i, neighbor - D)
                        R_i = min(R_i, neighbor + D)

                    # Check if the required interval is valid
                    if L_i > R_i:
                        return False # This -1 position cannot be filled
                    required_intervals.append([L_i, R_i])
            
            # This case is covered by max_fixed_diff check if has_missing is false.
            # If has_missing is true but required_intervals is empty,
            # it means -1s are not adjacent to any non-missing values.
            # Example: [-1, -1, 5, 6, -1, -1]. P={5, 6}.
            # Example: [-1, -1]. P={}. This is handled by all -1s case.
            # Example: [5, -1, -1, 6]. P={5, 6}. Required intervals for index 1 and 2.
            # If we reach here and required_intervals is empty, it shouldn't happen
            # unless the problem implies only -1s flanked by non--1s create requirements.
            # The problem implies *any* -1 can be replaced by x or y.
            # A -1 at index 0 needs replacement. If nums[1] is non--1, it constrains nums[0].
            # If nums[1] is -1, nums[0] and nums[1] must be |r0 - r1| <= D.
            # The code calculates Req_i only based on non--1 neighbors.
            # If nums = [-1, -1], Req for index 0 and 1 are [1, 10^9+7].
            # The constraint |r0 - r1| <= D is missing in Req_i calculation.
            # However, this constraint |x-y| <= D is checked implicitly by the two-value check.

            # Let's rely on the simplified checks derived from min_P and max_P.

            # Case 1: Single value k can cover all required intervals
            # This requires an intersection of all required intervals.
            # This seems stronger than required. Maybe it's sufficient that k covers P.
            # Let's use the simplified logic: Check single value cover based on P.
            overall_P_L = max(1, max_P - D)
            overall_P_R = min_P + D
            if overall_P_L <= overall_P_R:
                return True # There exists a value k in [overall_P_L, overall_P_R] that is <= D from all v in P

            # Case 2: Two values x, y can cover min_P and max_P respectively within distance D, and |x-y| <= D
            I_min_P_L = max(1, min_P - D)
            I_min_P_R = min_P + D
            I_max_P_L = max(1, max_P - D)
            I_max_P_R = max_P + D

            min_dist_between_min_max_intervals = max(0, I_max_P_L - I_min_P_R)
            if min_dist_between_min_max_intervals <= D:
                 return True # There exist x in [I_min_P_L, I_min_P_R] and y in [I_max_P_L, I_max_P_R] with |x-y| <= D
            
            return False # Neither single value cover for P nor the two value cover for min_P/max_P works.


        # Binary search for the minimum D
        low = max_fixed_diff
        # A safe upper bound: max possible difference (10^9). Max difference can be 10^9 - 1.
        # Or max value of nums if any non-missing exists.
        # If P exists, max_P - min_P can be up to 10^9.
        # Max possible difference is roughly max(max_fixed_diff, max_P - min_P).
        # A safe upper bound for D is 10^9.
        high = 10**9 # Upper bound for D
        ans = high 

        while low <= high:
            mid = low + (high - low) // 2
            if can(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1

        return ans