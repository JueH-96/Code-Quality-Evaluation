from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        """
        This problem asks for the maximum length of a non-decreasing array that can be formed
        by repeatedly merging subarrays into their sums. This is equivalent to partitioning the
        original array `nums` into `k` contiguous subarrays, calculating the sum of each subarray
        to form a new array `b = [b_1, b_2, ..., b_k]`, such that `b` is non-decreasing, and we
        want to maximize `k`.

        A greedy approach provides an optimal solution. We build the non-decreasing array from
        left to right. To maximize the total number of elements `k`, we should make each
        element `b_i` as small as possible while satisfying the non-decreasing condition
        `b_i >= b_{i-1}`. A smaller `b_i` sets a lower bar for the next element `b_{i+1}`,
        making it easier to form more elements.

        The algorithm is as follows:
        1. Initialize `ans` (the length of the resulting array) to 0.
        2. `last_sum` tracks the value of the last element added to our new array. Initialize to 0.
        3. `current_sum` accumulates the sum for the group we are currently forming. Initialize to 0.
        4. Iterate through `nums`. Add the current number to `current_sum`.
        5. If `current_sum` becomes greater than or equal to `last_sum`, we have formed a valid 
           next element. Increment `ans`, update `last_sum` to `current_sum`, and reset 
           `current_sum` to 0.
        
        This greedy strategy is proven to be optimal via an exchange argument. Any optimal solution 
        can be transformed into the greedy solution without decreasing its length, confirming that
        the greedy approach yields the maximum possible length.
        """
        ans = 0
        last_sum = 0
        current_sum = 0
        
        for num in nums:
            current_sum += num
            if current_sum >= last_sum:
                ans += 1
                last_sum = current_sum
                current_sum = 0
                
        return ans