import math
from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # ans_diff will store the changes in the total count of modifications
        # as X increases. Its size needs to be k+2 to handle indices up to k+1.
        # This allows us to mark the end of ranges correctly.
        ans_diff = [0] * (k + 2)
        
        # Iterate over n/2 pairs of (nums[i], nums[n-1-i])
        for i in range(n // 2):
            a = nums[i]
            b = nums[n - 1 - i]
            
            # Ensure a <= b for consistent calculation of min/max
            if a > b:
                a, b = b, a
            
            # m = min(a, b)
            # M = max(a, b)
            # diff = abs(a - b) = M - m
            m = a
            M = b
            diff = M - m
            
            # --- Contribution analysis for the current pair (a, b) ---
            
            # 1. Base cost:
            # Each pair initially contributes 2 changes for any X.
            # This is because we can always change both 'a' and 'b' to achieve X
            # (e.g., set a'=0, b'=X, assuming X <= k).
            # So, we add 2 to the count for all X from 0 to k.
            ans_diff[0] += 2
            ans_diff[k + 1] -= 2 # Mark the end of this universal range contribution (exclusive)

            # 2. Saving 1 change:
            # For certain X values, only 1 change is needed for this pair.
            # This occurs if X can be formed by abs(val - fixed_original_val),
            # where val is in the range [0, k].
            # This condition is met if X <= max(M, k - m).
            # Let R1 = max(M, k - m).
            # For X in the range [0, R1], we can achieve the target difference with 1 change.
            # This means we save 1 change compared to the base cost of 2.
            # So, we subtract 1 from the count for X in [0, R1].
            upper_bound_1_change = max(M, k - m)
            ans_diff[0] -= 1
            ans_diff[upper_bound_1_change + 1] += 1 # Mark the end of this 1-change saving range (exclusive)

            # 3. Saving another 1 change (total saving of 2):
            # If X is exactly the original difference (diff = M - m), then 0 changes are needed.
            # This is an additional saving of 1, on top of the 1 already saved
            # for the range [0, upper_bound_1_change] (where diff is always included).
            # So, we subtract another 1 specifically for X = diff.
            ans_diff[diff] -= 1
            ans_diff[diff + 1] += 1 # Mark the end of this specific 0-change saving for X=diff (exclusive)

        # After processing all pairs, compute the prefix sums to get the total changes for each X.
        min_overall_changes = float('inf')
        current_changes = 0 # This will hold the total changes for the current X
        
        # Iterate from X = 0 to k
        for x in range(k + 1):
            current_changes += ans_diff[x]
            min_overall_changes = min(min_overall_changes, current_changes)
            
        return min_overall_changes