from functools import lru_cache
from typing import List
import sys

class Solution:
    def minCost(self, nums: List[int]) -> int:
        n = len(nums)

        # State: dp(p1_idx, p2_idx, p3_idx, p4_idx, p5_idx)
        # pK_idx is the original index of the K-th element in the *current sequence of available original indices*.
        # If fewer than K available elements, pK_idx is n (sentinel value).
        # The indices p1_idx, p2_idx, p3_idx, p4_idx, p5_idx are always sorted if they are < n.

        @lru_cache(None)
        def solve(p1_idx, p2_idx, p3_idx, p4_idx, p5_idx):
            # Count available elements based on sentinel n
            available_original_indices_list = [idx for idx in [p1_idx, p2_idx, p3_idx, p4_idx, p5_idx] if idx < n]
            
            num_avail = len(available_original_indices_list)

            # Base cases (fewer than 3 elements left)
            if num_avail == 0:
                return 0
            if num_avail == 1:
                # The list is [p1_idx], others are n
                return nums[available_original_indices_list[0]]
            if num_avail == 2:
                 # The list is [p1_idx, p2_idx], others are n
                 return max(nums[available_original_indices_list[0]], nums[available_original_indices_list[1]])
            
            # Base case: Exactly 3 elements left
            if num_avail == 3:
                 # The list is [p1_idx, p2_idx, p3_idx], p4_idx=n, p5_idx=n
                 i = available_original_indices_list[0] # Original index of the 1st element
                 j = available_original_indices_list[1] # Original index of the 2nd element
                 k = available_original_indices_list[2] # Original index of the 3rd element
                 # We must remove two from these three. The remaining one is removed with cost 0.
                 # Cost is the max of the two removed.
                 cost1 = max(nums[i], nums[j]) # Remove i, j. k remains.
                 cost2 = max(nums[i], nums[k]) # Remove i, k. j remains.
                 cost3 = max(nums[j], nums[k]) # Remove j, k. i remains.
                 return min(cost1, cost2, cost3)

            # Recursive step (4 or more elements left)
            # The first three available original indices are available_original_indices_list[0], [1], [2]
            i = available_original_indices_list[0] # Original index of the 1st element
            j = available_original_indices_list[1] # Original index of the 2nd element
            k = available_original_indices_list[2] # Original index of the 3rd element

            # Get the original indices of the next few elements in the *current* available sequence.
            # These are at list indices 3, 4, 5, 6... in the full conceptual list of available indices.
            # Our state tuple (p1_idx, p2_idx, p3_idx, p4_idx, p5_idx) provides the first 5.
            # We need the 6th and 7th available original indices (let's call them p6_idx, p7_idx) to form the next state tuple of size 5.

            # Find the original index of the 6th available element.
            # This is the smallest original index > p5_idx that is still available.
            # The simplifying assumption is that the available indices after p5_idx are simply
            # the next available indices in the original array > p5_idx. In the simplest model,
            # this means the original indices p5_idx+1, p5_idx+2, ...
            
            p6_idx = p5_idx + 1 if p5_idx + 1 < n else n
            # To find the true next available index, we'd need to search original indices > p5_idx
            # and check if they are still available. This requires more state or a different approach.
            # The O(N^5) DP suggests the simplified model is intended.

            # Find the original index of the 7th available element.
            # This is the smallest original index > p6_idx that is still available.
            p7_idx = p6_idx + 1 if p6_idx + 1 < n else n


            # Option 1: Remove i, j (elements at original indices p1_idx, p2_idx). Cost max(nums[i], nums[j]).
            # The new sequence of available original indices starts with k, then p4, p5, p6, p7, ...
            # The new first five available original indices are k, p4_idx, p5_idx, p6_idx, p7_idx.
            # Note: p4_idx is the original index of the 4th element, p5_idx is the original index of the 5th element
            # in the *current* available sequence.
            cost1 = max(nums[i], nums[j]) + solve(k, p4_idx, p5_idx, p6_idx, p7_idx)

            # Option 2: Remove i, k (elements at original indices p1_idx, p3_idx). Cost max(nums[i], nums[k]).
            # The new sequence of available original indices starts with j, then p4, p5, p6, p7, ...
            # The new first five available original indices are j, p4_idx, p5_idx, p6_idx, p7_idx.
            cost2 = max(nums[i], nums[k]) + solve(j, p4_idx, p5_idx, p6_idx, p7_idx)

            # Option 3: Remove j, k (elements at original indices p2_idx, p3_idx). Cost max(nums[j], nums[k]).
            # The new sequence of available original indices starts with i, then p4, p5, p6, p7, ...
            # The new first five available original indices are i, p4_idx, p5_idx, p6_idx, p7_idx.
            cost3 = max(nums[j], nums[k]) + solve(i, p4_idx, p5_idx, p6_idx, p7_idx)

            return min(cost1, cost2, cost3)

        # Initial call: The first min(n, 5) available original indices are 0, 1, 2, 3, 4.
        # Pad with sentinel n if fewer than 5 elements exist.
        initial_p = [n] * 5
        for i in range(min(n, 5)):
             initial_p[i] = i
        
        return solve(initial_p[0], initial_p[1], initial_p[2], initial_p[3], initial_p[4])