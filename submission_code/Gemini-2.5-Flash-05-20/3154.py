import math
from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)

        # The problem constraints state 3 <= nums.length <= 100.
        # This means there will always be at least one valid triplet (i, j, k).

        # 1. Precompute max_prefix:
        # max_prefix[i] stores the maximum value in nums[0...i].
        # This helps in finding the maximum nums[x] for x < j.
        max_prefix = [0] * n
        max_prefix[0] = nums[0]
        for i in range(1, n):
            max_prefix[i] = max(max_prefix[i-1], nums[i])

        # 2. Precompute max_from_right:
        # max_from_right[i] stores the maximum value in nums[i...n-1].
        # This helps in finding the maximum nums[x] for x > j.
        # Specifically, for a given j, the max nums[k] where k > j will be max_from_right[j+1].
        max_from_right = [0] * n
        max_from_right[n-1] = nums[n-1] # Base case for the last element
        for i in range(n - 2, -1, -1): # Iterate from n-2 down to 0
            max_from_right[i] = max(max_from_right[i+1], nums[i])

        # Initialize the maximum triplet value found so far to negative infinity.
        # This ensures that any valid triplet value (positive, zero, or negative)
        # will be greater than or equal to this initial value.
        max_triplet_val = -math.inf

        # 3. Iterate through all possible middle indices 'j'.
        # 'j' must satisfy i < j < k.
        # The smallest 'j' can be is 1 (e.g., i=0, j=1, k=2).
        # The largest 'j' can be is n-2 (e.g., i=n-3, j=n-2, k=n-1).
        for j in range(1, n - 1):
            # For the current 'j', we need to find the optimal 'i' (where i < j)
            # and the optimal 'k' (where k > j) to maximize the triplet value.
            
            # 'val_i' is the maximum element encountered before index 'j'.
            # This is obtained from the precomputed max_prefix array.
            val_i = max_prefix[j-1]

            # 'val_k' is the maximum element encountered after index 'j'.
            # This is obtained from the precomputed max_from_right array.
            # Since nums[k] is always positive (1 <= nums[k] <= 10^6),
            # to maximize the product (val_i - nums[j]) * val_k, we always choose the largest possible val_k.
            val_k = max_from_right[j+1]

            # Calculate the triplet value for this 'j' using the optimal 'i' and 'k'.
            current_value = (val_i - nums[j]) * val_k
            
            # Update the overall maximum triplet value found.
            max_triplet_val = max(max_triplet_val, current_value)
        
        # According to the problem statement:
        # "If all such triplets have a negative value, return 0."
        # This means the final result should be the maximum of 0 and the actual maximum triplet value found.
        return max(0, max_triplet_val)