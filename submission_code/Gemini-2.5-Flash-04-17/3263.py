from typing import List
import sys

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        """
        Calculates the minimum possible sum of costs when dividing the array
        into three disjoint contiguous subarrays. The cost of a subarray
        is the value of its first element.

        Args:
            nums: A list of integers of length n (3 <= n <= 50).

        Returns:
            The minimum possible sum of the costs of the three subarrays.
        """
        # We divide nums into 3 contiguous subarrays.
        # Let the division points be just before index i and just before index j,
        # where 0 < i < j < n.
        # Subarray 1: nums[0 : i]. Its first element is nums[0]. Cost is nums[0].
        # Subarray 2: nums[i : j]. Its first element is nums[i]. Cost is nums[i].
        # Subarray 3: nums[j : n]. Its first element is nums[j]. Cost is nums[j].
        # For these to be valid, non-empty subarrays, the indices must satisfy:
        # 1. The first subarray `nums[0:i]` must be non-empty, so `i > 0`. Minimum `i` is 1.
        # 2. The second subarray `nums[i:j]` must be non-empty, so `j > i`. Minimum `j` is `i + 1`.
        # 3. The third subarray `nums[j:n]` must be non-empty, so `n > j`. Maximum `j` is `n - 1`.
        # Combining these, the valid range for the start indices of the second and third
        # subarrays (i and j) is `1 <= i < j <= n - 1`.

        # The total cost for a specific division defined by `i` and `j` is `nums[0] + nums[i] + nums[j]`.
        # We want to find the minimum possible total cost by choosing optimal `i` and `j`
        # such that `1 <= i < j <= n - 1`.

        # The term `nums[0]` is the cost of the first subarray and is fixed for all possible divisions
        # starting at index 0.
        # To minimize the total cost, we need to minimize the sum `nums[i] + nums[j]`
        # over all valid pairs of indices `i` and `j` from the set `{1, 2, ..., n-1}`
        # such that `i < j`.

        # This is equivalent to finding the two smallest values among the elements
        # `nums[1], nums[2], ..., nums[n-1]`. Let these two smallest values be `val1` and `val2`.
        # The minimum possible sum `nums[i] + nums[j]` will be `val1 + val2`.

        # The minimum total cost is then `nums[0] + val1 + val2`.

        n = len(nums)

        # We need to find the two smallest values among nums[1]...nums[n-1].
        # We can iterate through this part of the array once to find the two smallest.
        # Since n >= 3, the subarray nums[1:] contains at least 2 elements (indices 1 and 2),
        # so we are guaranteed to find two smallest values.

        # Initialize variables to track the two smallest values found so far in nums[1:].
        # Initialize them to a value larger than any possible element in nums (max is 50).
        # sys.maxsize is a safe way to get a very large integer.
        smallest_val_1 = sys.maxsize
        smallest_val_2 = sys.maxsize

        # Iterate through the elements from index 1 to n-1.
        # These are the potential first elements (and thus costs) of the
        # second and third subarrays.
        for k in range(1, n):
            current_val = nums[k]

            # Check if the current value is smaller than the current smallest found.
            if current_val < smallest_val_1:
                # If it is, the previous smallest becomes the second smallest,
                # and the current value becomes the new smallest.
                smallest_val_2 = smallest_val_1
                smallest_val_1 = current_val
            # Otherwise, check if the current value is smaller than the current second smallest found.
            elif current_val < smallest_val_2:
                # If it is, the current value becomes the new second smallest.
                smallest_val_2 = current_val

        # The minimum total cost is the cost of the first subarray (nums[0])
        # plus the costs of the second and third subarrays (the two smallest
        # values found among nums[1]...nums[n-1]).
        min_total_cost = nums[0] + smallest_val_1 + smallest_val_2

        return min_total_cost