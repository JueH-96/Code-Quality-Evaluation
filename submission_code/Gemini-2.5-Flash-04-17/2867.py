from typing import List

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        # Find the indices of all ones in the array.
        ones_indices = [i for i, num in enumerate(nums) if num == 1]

        # If there are no ones, it's impossible to split the array into good subarrays,
        # because a good subarray must contain exactly one '1'.
        if not ones_indices:
            return 0

        # If there is exactly one '1', the entire array must form a single good subarray.
        # There is only one way to do this (by not splitting at all).
        # The loop below will handle this case correctly as well, as the range will be empty.
        # This explicit check is not strictly necessary but can make the logic clearer
        # or handle the m=1 case directly if preferred. Let's rely on the loop.

        MOD = 10**9 + 7
        
        # If there are 'm' ones at indices idx_1, idx_2, ..., idx_m, any valid split
        # must consist of exactly 'm' good subarrays s_1, s_2, ..., s_m.
        # s_1 must contain the first '1' (at idx_1), s_2 must contain the second '1'
        # (at idx_2), and so on, up to s_m containing the last '1' (at idx_m).
        # The splits must occur between consecutive ones.
        # Specifically, the split between subarray s_j (containing idx_j) and
        # subarray s_{j+1} (containing idx_{j+1}) must happen at some index k
        # such that the subarray ending at k contains idx_j and the subarray
        # starting at k+1 contains idx_{j+1}. This implies idx_j <= k < idx_{j+1}.
        # The number of choices for the split index k between the j-th one (at index ones_indices[j])
        # and the (j+1)-th one (at index ones_indices[j+1]) is the number of integers
        # in the range [ones_indices[j], ones_indices[j+1] - 1], which is
        # (ones_indices[j+1] - 1) - ones_indices[j] + 1 = ones_indices[j+1] - ones_indices[j].
        # This is the distance between the indices of consecutive ones.

        # The total number of ways to split the array is the product of the number
        # of choices for each split point between consecutive ones.

        result = 1
        # We iterate through pairs of consecutive ones (i-th one and (i+1)-th one).
        # The loop runs from i=0 to len(ones_indices) - 2.
        for i in range(len(ones_indices) - 1):
            # Calculate the distance between the current '1' and the next '1'.
            # This distance is the number of valid places to make a split
            # between these two '1's.
            distance = ones_indices[i+1] - ones_indices[i]
            
            # Multiply the result by the number of choices for this split point,
            # taking the modulo at each step to prevent overflow.
            result = (result * distance) % MOD

        return result