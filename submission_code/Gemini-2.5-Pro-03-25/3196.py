import math
from typing import List

class Solution:
    """
    Finds the maximum frequency score achievable by modifying array elements.
    The total modification cost must not exceed k.

    The problem asks for the maximum possible frequency of an element in the array `nums` 
    after performing at most k operations. Each operation allows increasing or decreasing 
    an element `nums[i]` by 1. The total number of operations (sum of absolute changes) 
    must not exceed `k`.

    To achieve a frequency `f` for some target value `x`, we need to select `f` elements 
    from the original array and modify them to become `x`. The cost of changing `nums[i]` to `x`
    is `|nums[i] - x|`. The total cost for a chosen set of `f` elements is `sum(|nums[i] - x|)`.
    To minimize this cost for a fixed set of `f` elements, the optimal target value `x` is 
    the median of these `f` elements.

    It can be shown that the optimal strategy involves choosing a contiguous subarray of 
    the sorted version of `nums`. Let the sorted array be `s_nums`. We want to find the 
    largest `f` such that there exists a subarray `s_nums[i], ..., s_nums[i+f-1]` whose elements 
    can be made equal to their median with a total cost less than or equal to `k`.

    We can use a sliding window approach on the sorted array `s_nums`. We maintain a window 
    `[left, right]` and calculate the cost to make all elements in `s_nums[left...right]` 
    equal to their median. We expand the window by incrementing `right`. If the cost exceeds `k`,
    we shrink the window by incrementing `left` until the cost is within the budget `k`.
    We keep track of the maximum window size encountered that satisfies the cost constraint.

    The cost calculation can be done efficiently using prefix sums.
    Let `P` be the prefix sum array of `s_nums`.
    For a subarray `s_nums[i...j]`, the length is `f = j - i + 1`.
    The median index relative to `i` is `k' = floor((f-1)/2)`.
    The absolute median index is `mid_idx = i + k'`.
    The median value is `m = s_nums[mid_idx]`.
    The cost is `sum_{p=i}^{j} |s_nums[p] - m|`.
    This can be calculated as `(sum_{p=i}^{mid_idx} (m - s_nums[p])) + (sum_{p=mid_idx+1}^{j} (s_nums[p] - m))`.
    Using prefix sums:
    Cost = `( (mid_idx - i + 1) * m - (P[mid_idx+1] - P[i]) ) + ( (P[j+1] - P[mid_idx+1]) - (j - mid_idx) * m )`.
    """
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        """
        Calculates the maximum frequency score using a sliding window approach on the sorted array.

        Args:
            nums: A list of integers.
            k: The maximum allowed total cost of operations.

        Returns:
            The maximum achievable frequency score.
        """
        
        n = len(nums)
        if n == 0:
            return 0
        
        # Sort the array to enable processing contiguous subarrays.
        nums.sort()
        
        # Calculate prefix sums for efficient cost calculation within subarrays.
        # Python's arbitrary precision integers handle potentially large sums and k.
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]
            
        # Helper function to calculate the minimum cost to make all elements
        # in the subarray nums[i...j] equal. The target value is their median.
        # This function runs in O(1) time due to prefix sums.
        def calculate_cost(i: int, j: int) -> int:
            """
            Calculates the minimum cost to make elements nums[i..j] equal using median as target.
            Assumes nums is sorted. Uses prefix sums for O(1) calculation.
            """
            # If the window is invalid or empty, cost is 0.
            if i > j:
                return 0
            
            length = j - i + 1
            # Calculate median index and value
            # For odd length `f`, median is at index `(f-1)/2`.
            # For even length `f`, median can be any value between elements at indices `f/2 - 1` and `f/2`.
            # Choosing `nums[i + (length-1)//2]` as target `m` minimizes sum of absolute differences.
            k_prime = (length - 1) // 2  # Median index relative to start `i`
            median_idx = i + k_prime      # Absolute index in `nums`
            median_val = nums[median_idx]
            
            # Cost calculation using prefix sums derived from the definition sum |x_p - median|
            
            # Cost for elements left of or at the median: sum (median_val - nums[p]) for p in [i..median_idx]
            left_sum = prefix_sum[median_idx + 1] - prefix_sum[i]
            left_count = median_idx - i + 1
            left_cost = left_count * median_val - left_sum
            
            # Cost for elements right of the median: sum (nums[p] - median_val) for p in [median_idx+1..j]
            right_sum = prefix_sum[j + 1] - prefix_sum[median_idx + 1]
            right_count = j - median_idx
            right_cost = right_sum - right_count * median_val
            
            # Total cost is the sum of costs for left and right parts
            cost = left_cost + right_cost
            return cost

        # Sliding window approach:
        # `left` pointer marks the start of the window
        # `right` pointer marks the end of the window
        left = 0       
        max_freq = 0   # Stores the maximum frequency (window length) found so far

        # Iterate with the `right` pointer to expand the window
        for right in range(n):
            # Calculate the cost for the current window [left...right]
            current_cost = calculate_cost(left, right)

            # If the current window's cost exceeds the budget `k`,
            # shrink the window from the left by incrementing `left`.
            # Repeat until the cost is within the budget.
            while current_cost > k:
                 # Increment left pointer to shrink window
                 left += 1
                 # If window becomes empty (left > right), cost will be 0. 
                 # The helper `calculate_cost` handles this.
                 current_cost = calculate_cost(left, right) # Recalculate cost for the new window

            # After ensuring the cost is <= k, the current window [left...right] is valid.
            # Update the maximum frequency found so far with the current window's length.
            current_length = right - left + 1
            max_freq = max(max_freq, current_length)
            
        # Return the maximum frequency found.
        return max_freq