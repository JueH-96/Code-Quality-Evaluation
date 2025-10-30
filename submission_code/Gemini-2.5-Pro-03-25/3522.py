import math # Unused import, can be removed but doesn't hurt
from typing import List 
import itertools # Used for accumulate function

class Solution:
    """
    Calculates the power of all subarrays of size k in the given array nums.

    The power of an array is defined as:
    - Its maximum element if all of its elements are consecutive and sorted in ascending order.
    - -1 otherwise.

    This solution uses a prefix sum approach for O(n) time complexity.
    """
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        """
        Finds the power for each subarray of nums of size k.

        Args:
            nums: The input list of integers. Constraints: 1 <= n == nums.length <= 500, 1 <= nums[i] <= 10^5.
            k: The size of the subarrays. Constraints: 1 <= k <= n.

        Returns:
            A list of integers of size n - k + 1, where the i-th element 
            is the power of the subarray nums[i:i+k].
        """
        n = len(nums)
        
        # Constraints guarantee 1 <= k <= n and n >= 1.

        # Step 1: Precompute the difference check array `d`.
        # `d[p]` will be 1 if nums[p+1] is exactly one greater than nums[p], and 0 otherwise.
        # This array helps check the consecutive and sorted condition.
        # The size of `d` will be n-1.
        # If n=1 (which implies k=1), the range(n-1) is empty, so d will be empty.
        d = [0] * (n - 1)
        for p in range(n - 1):
            if nums[p+1] == nums[p] + 1:
                d[p] = 1
        
        # Step 2: Calculate prefix sums of the `d` array.
        # `prefix_sum[p]` stores the sum of `d[0]` up to `d[p-1]`.
        # `itertools.accumulate` with `initial=0` efficiently computes this.
        # The resulting `prefix_sum` list will have size n. `prefix_sum[0]` will be 0.
        # If n=1, d is empty, accumulate([ ], initial=0) results in [0]. prefix_sum = [0], size 1=n. This is correct.
        prefix_sum = list(itertools.accumulate(d, initial=0))

        results = []
        # Step 3: Iterate through all possible starting indices `i` for subarrays of size k.
        # The loop runs from i = 0 up to n - k.
        for i in range(n - k + 1):
            # Step 4: Check if the subarray nums[i:i+k] is consecutive and sorted.
            # This condition holds if and only if all the k-1 differences within the subarray are exactly 1.
            # In terms of the `d` array, this means d[p] must be 1 for all p from i to i+k-2.
            
            # We can check this efficiently using the prefix sums. The sum of `d[i]` through `d[i+k-2]`
            # must be equal to the number of terms in this range, which is (i+k-2) - i + 1 = k-1.
            
            # Calculate the sum of d[i]...d[i+k-2] using prefix sums:
            # range_sum = prefix_sum[end_index + 1] - prefix_sum[start_index]
            # Here, start_index = i, end_index = i+k-2
            # range_sum = prefix_sum[(i+k-2) + 1] - prefix_sum[i]
            # range_sum = prefix_sum[i + k - 1] - prefix_sum[i]
            range_sum = prefix_sum[i + k - 1] - prefix_sum[i]
            
            # The number of terms we are summing in `d` is k-1.
            # If k=1, the number of terms is 0, and the range is empty.
            # In this case, range_sum = prefix_sum[i] - prefix_sum[i] = 0.
            # The condition check `range_sum == k - 1` becomes `0 == 1 - 1`, which is true.
            # So, the logic correctly handles the k=1 case.
            if range_sum == k - 1:
                # If the sum is k-1, it means all k-1 differences were 1.
                # The subarray is consecutive and sorted.
                # The power is the maximum element, which is the last element in the sorted subarray.
                # Maximum element is nums[i + k - 1]. (For k=1, this is nums[i]).
                results.append(nums[i + k - 1])
            else:
                # If the sum is not k-1, it means at least one difference was not 1.
                # The subarray is not consecutive and sorted according to the definition.
                # The power is -1.
                results.append(-1)
                
        # Step 5: Return the array of calculated powers.
        return results