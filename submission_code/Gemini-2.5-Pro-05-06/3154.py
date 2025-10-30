from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_overall_value = 0

        # Constraints: 3 <= nums.length <= 100. nums[i] are positive.
        # This ensures n >= 3, so nums[0] exists and loops below are valid.

        # max_val_for_num_i will store max(nums[p] for p < current_j_idx)
        # It's initialized with nums[0] because when j_idx = 1 (first iteration),
        # p must be 0, so max(nums[p] for p < 1) is nums[0].
        max_val_for_num_i = nums[0] 

        # j_idx is the index for the middle element nums[j].
        # It ranges from 1 to n-2 (inclusive for n-2).
        # So, i (implicitly chosen for max_val_for_num_i) can range from 0 to j_idx-1.
        # And k (k_idx) can range from j_idx+1 to n-1.
        for j_idx in range(1, n - 1):
            # At this point, max_val_for_num_i represents max(nums[0], ..., nums[j_idx-1]).
            # This is the optimal choice for nums[i] if we fix nums[j] = nums[j_idx]
            # and want to maximize (nums[i] - nums[j]).
            
            # We only care if (max_val_for_num_i - nums[j_idx]) is positive,
            # because nums[k_idx] is positive. If (max_val_for_num_i - nums[j_idx]) <= 0,
            # the triplet value will be <= 0. Since max_overall_value is initialized to 0
            # and we only take strictly larger values, non-positive triplet values
            # would not update max_overall_value.
            if max_val_for_num_i > nums[j_idx]:
                diff = max_val_for_num_i - nums[j_idx]
                
                # k_idx is the index for the third element nums[k].
                # It ranges from j_idx+1 to n-1.
                for k_idx in range(j_idx + 1, n):
                    current_triplet_value = diff * nums[k_idx]
                    if current_triplet_value > max_overall_value:
                        max_overall_value = current_triplet_value
            
            # Update max_val_for_num_i for the *next* iteration of j_idx.
            # The current nums[j_idx] becomes a candidate for the first element
            # (i.e., part of max(nums[0...next_j_idx-1])) when j_idx advances.
            if nums[j_idx] > max_val_for_num_i:
                max_val_for_num_i = nums[j_idx]
            
        return max_overall_value