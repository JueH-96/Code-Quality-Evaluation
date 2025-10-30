from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        """
        Calculates the maximum possible bitwise OR of the array elements
        after applying the multiplication-by-2 operation at most k times.

        The core idea is based on the realization that to maximize the bitwise OR, 
        it is always optimal to apply all k operations to a single element. 
        Multiplying an element `nums[i]` by `2^k` (equivalent to `nums[i] << k`) shifts its bits k positions to the left.
        This potentially introduces set bits at higher positions, which contributes significantly to the OR sum. 
        Distributing the k operations among multiple elements results in smaller shifts for each element,
        which is generally less effective for maximizing the overall OR value compared to maximizing the shift for one element.

        The algorithm proceeds as follows:
        1. Calculate prefix OR sums: `prefix_or[i]` stores the bitwise OR of `nums[0]` through `nums[i]`.
        2. Calculate suffix OR sums: `suffix_or[i]` stores the bitwise OR of `nums[i]` through `nums[n-1]`.
        3. Iterate through each element `nums[i]`. For each `i`, consider the scenario where all k operations are applied to `nums[i]`.
           The modified value of `nums[i]` becomes `nums[i] * (2^k)`.
           The overall bitwise OR for this scenario is calculated as:
           `(OR of elements before i) | (nums[i] * 2^k) | (OR of elements after i)`.
           The OR of elements before `i` is `prefix_or[i-1]` (or 0 if `i=0`).
           The OR of elements after `i` is `suffix_or[i+1]` (or 0 if `i=n-1`).
        4. Keep track of the maximum OR value found across all scenarios (for all `i`).
        5. Return the overall maximum OR value.

        This approach takes O(N) time for prefix and suffix calculations and O(N) time for the final iteration,
        resulting in a total time complexity of O(N). The space complexity is O(N) for storing prefix and suffix OR sums.

        Args:
          nums: A list of 0-indexed integers. Constraints: 1 <= nums[i] <= 10^9.
          k: The maximum number of operations allowed. Constraints: 1 <= k <= 15.

        Returns:
          The maximum possible bitwise OR value attainable.
        """
        n = len(nums)
        
        # Calculate prefix OR sums: prefix_or[i] = nums[0] | ... | nums[i]
        # This array helps quickly find the OR sum of elements up to index i-1.
        prefix_or = [0] * n
        prefix_or[0] = nums[0]
        for i in range(1, n):
            prefix_or[i] = prefix_or[i-1] | nums[i]
            
        # Calculate suffix OR sums: suffix_or[i] = nums[i] | ... | nums[n-1]
        # This array helps quickly find the OR sum of elements from index i+1 onwards.
        suffix_or = [0] * n
        suffix_or[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            suffix_or[i] = suffix_or[i+1] | nums[i]
            
        max_or_value = 0
        
        # Iterate through each element nums[i], considering the case where all k operations are applied to it.
        for i in range(n):
            # Get the OR sum of elements strictly before index i.
            # If i is 0, there are no elements before it, so the prefix OR contribution is 0.
            p_or = prefix_or[i-1] if i > 0 else 0
            
            # Get the OR sum of elements strictly after index i.
            # If i is n-1, there are no elements after it, so the suffix OR contribution is 0.
            s_or = suffix_or[i+1] if i < n-1 else 0
            
            # This computes the bitwise OR of all elements *except* the original nums[i].
            # It represents the base OR value contributed by other elements.
            or_without_i = p_or | s_or
            
            # Calculate the value of nums[i] after applying k operations.
            # Multiplying by 2^k is equivalent to left bit-shifting by k positions.
            current_num_modified = nums[i] << k
            
            # Calculate the total OR value for this specific scenario (operations applied only to nums[i]).
            # This combines the OR of other elements with the modified nums[i].
            current_total_or = or_without_i | current_num_modified
            
            # Update the overall maximum OR value found so far.
            max_or_value = max(max_or_value, current_total_or)
            
        return max_or_value